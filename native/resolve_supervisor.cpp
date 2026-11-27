// Isolate the standalone CLI's Python computation until its deadline. The
// child parses arguments and renders records using the regular Atlas entry.
// The search publishes its progress in a file that this process names. When
// the time budget of a quick search ends, the worker is stopped and the
// regular entry is started again to answer: it reads that file to say what
// the search was doing.
#include <sys/prctl.h>
#include <sys/signalfd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

using Clock = std::chrono::steady_clock;

static int python(char **args, const char *control) {
    std::string entry;
    std::vector<char *> command{const_cast<char *>("python3")};
    if (control) {
        std::error_code error;
        entry = (std::filesystem::canonical(args[0], error).parent_path() / "engine" / "supervised_cli.py").string();
        if (error) return 127;
        command.push_back(entry.data());
        command.push_back(const_cast<char *>(control));
    } else {
        command.push_back(args[0]);
    }
    for (char **arg = args + 1; *arg; ++arg) command.push_back(*arg);
    command.push_back(nullptr);
    execvp(command[0], command.data());
    perror("python3");
    return 127;
}

// Processes started by exhaustive search can create their own sessions.
// Once the main worker exits, Linux reparents its remaining descendants to
// this single-threaded subreaper. Kill and reap each generation.
static void cleanup(pid_t child, bool reaped) {
    if (!reaped) {
        // An active worker or waitable zombie reserves this PID/PGID.
        kill(-child, SIGKILL);
        kill(child, SIGKILL);
        while (waitpid(child, nullptr, 0) < 0 && errno == EINTR) {}
    }
    for (;;) {
        int status;
        pid_t waited = waitpid(-1, &status, WNOHANG);
        if (waited > 0 || (waited < 0 && errno == EINTR)) continue;
        if (waited < 0) return;
        std::ifstream children("/proc/thread-self/children");
        pid_t pid;
        while (children >> pid) kill(pid, SIGKILL);
        if (waitpid(-1, &status, 0) < 0 && errno != EINTR) return;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    // On hosts without a usable Linux process tree, use the existing
    // Python supervisor. No worker has been started at this point.
    if (access("/proc/thread-self/children", R_OK) || prctl(PR_SET_CHILD_SUBREAPER, 1))
        return python(argv + 1, nullptr);
    sigset_t signals, old;
    sigemptyset(&signals);
    for (int sig : {SIGINT, SIGTERM, SIGHUP, SIGCHLD}) sigaddset(&signals, sig);
    if (sigprocmask(SIG_BLOCK, &signals, &old)) return python(argv + 1, nullptr);
    int notices = signalfd(-1, &signals, SFD_CLOEXEC);
    int control[2];
    if (notices < 0 || pipe(control)) {
        if (notices >= 0) close(notices);
        sigprocmask(SIG_SETMASK, &old, nullptr);
        return python(argv + 1, nullptr);
    }
    // The file belongs to this run unless the caller already named one.
    std::string progress;
    if (!getenv("ATLAS_PROGRESS_FILE")) {
        const char *directory = getenv("TMPDIR");
        std::string name = std::string(directory && *directory ? directory : "/tmp") + "/atlas-progress-XXXXXX";
        const int handle = mkstemp(name.data());
        if (handle >= 0) {
            close(handle);
            progress = name;
            setenv("ATLAS_PROGRESS_FILE", progress.c_str(), 1);
            setenv("ATLAS_PROGRESS_OWNED", "1", 1);
        }
    }
    pid_t child = fork();
    if (child == 0) {
        close(control[0]);
        if (setsid() < 0) _exit(127);
        sigprocmask(SIG_SETMASK, &old, nullptr);
        _exit(python(argv + 1, std::to_string(control[1]).c_str()));
    }
    close(control[1]);
    if (child < 0) {
        close(control[0]);
        close(notices);
        sigprocmask(SIG_SETMASK, &old, nullptr);
        if (!progress.empty()) {
            unlink(progress.c_str());
            unsetenv("ATLAS_PROGRESS_FILE");
            unsetenv("ATLAS_PROGRESS_OWNED");
        }
        return python(argv + 1, nullptr);
    }

    fcntl(control[0], F_SETFL, O_NONBLOCK);
    pollfd fds[] = {{notices, POLLIN, 0}, {control[0], POLLIN, 0}};
    auto deadline = Clock::time_point::max();
    bool started = false, finished = false, reaped = false;
    int status = 0, interrupted = 0;
    const char *failure = nullptr;
    for (;;) {
        int timeout = -1;
        if (deadline != Clock::time_point::max()) {
            const auto left = std::chrono::ceil<std::chrono::milliseconds>(deadline - Clock::now()).count();
            timeout = static_cast<int>(std::clamp<decltype(left)>(left, 0, INT_MAX));
        }
        int ready = poll(fds, 2, timeout);
        if (ready < 0 && errno == EINTR) continue;
        if (ready < 0) { failure = "stopped"; break; }
        if (fds[0].revents & POLLIN) {
            signalfd_siginfo info{};
            if (read(notices, &info, sizeof(info)) == sizeof(info) && info.ssi_signo != SIGCHLD) {
                interrupted = static_cast<int>(info.ssi_signo);
                break;
            }
        }
        // Peek, leaving the PID reserved until any remaining children
        // from the worker's session have been signalled.
        siginfo_t exit_info{};
        const int result = waitid(P_PID, child, &exit_info, WEXITED | WNOHANG | WNOWAIT);
        if (fds[1].fd >= 0 && (exit_info.si_pid == child || fds[1].revents & (POLLIN | POLLHUP))) {
            // Each start notice is one atomic pipe write. The two notices
            // together are at most ten bytes, including the quick interval.
            char messages[10];
            ssize_t size = read(control[0], messages, sizeof(messages));
            if (size > 0) {
                for (ssize_t i = 0; i < size; ++i) {
                    if (messages[i] == 'E') started = true;
                    if (messages[i] == 'Q' && size - i >= 9) {
                        uint64_t ms;
                        std::memcpy(&ms, messages + i + 1, sizeof(ms));
                        deadline = Clock::now() + std::chrono::milliseconds(ms);
                        started = true;
                        i += 8;
                    } else if (messages[i] == 'D') {
                        finished = true;
                        deadline = Clock::time_point::max();
                    }
                }
            } else if (size == 0) {
                fds[1].fd = -1;
            }
        }
        if (exit_info.si_pid == child) {
            kill(-child, SIGKILL);
            while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
            reaped = true;
            if (started && !finished) failure = "stopped";
            break;
        }
        if (result < 0 && errno != EINTR) { failure = "stopped"; break; }
        if (Clock::now() >= deadline) { failure = "timeout"; break; }
    }

    cleanup(child, reaped);
    close(control[0]);
    close(notices);
    sigprocmask(SIG_SETMASK, &old, nullptr);
    if (interrupted) {
        if (!progress.empty()) unlink(progress.c_str());
        if (interrupted == SIGINT) std::fputs("Resolution interrupted.\n", stderr);
        return 128 + interrupted;
    }
    // No record is printed before the worker sends its completion notice.
    // Let the regular CLI handle its original arguments and report failure.
    // It reads the progress file and removes it.
    if (failure) return python(argv + 1, failure);
    if (!progress.empty()) unlink(progress.c_str());
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 2;
}
