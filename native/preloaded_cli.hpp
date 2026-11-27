// Connect a standalone Linux supervisor to an already preloaded Atlas engine.
// A worker can only begin after both parties commit to the handoff. A missing,
// busy or stale server costs one local connection attempt.
#pragma once
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <poll.h>
#include <fcntl.h>
#include <cstddef>
#include <cstdlib>
#include <dirent.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

namespace atlas_cli {
// Preserve additional inheritable descriptors by keeping those calls local.
inline bool only_cli_descriptors(int input, int output) {
    DIR *dir = opendir("/proc/self/fd");
    if (!dir) return false;
    bool compatible = true;
    while (dirent *entry = readdir(dir)) {
        char *end = nullptr;
        int fd = static_cast<int>(strtol(entry->d_name, &end, 10));
        if (*end || fd < 3 || fd == input || fd == output || fd == dirfd(dir)) continue;
        if (fcntl(fd, F_GETFD) == 0) { compatible = false; break; }
    }
    closedir(dir);
    return compatible;
}

inline int borrow(char **args, const int control[2]) {
    const char *mode = getenv("ATLAS_PRELOADED_CLI");
    if (mode && strcmp(mode, "0") == 0) return -1;
    struct stat entry{};
    if (stat(args[0], &entry) != 0) return -1;
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    int count = std::snprintf(address.sun_path + 1, sizeof(address.sun_path) - 1,
                              "atlas.cli.1.%u.%llu.%llu", unsigned(geteuid()),
                              (unsigned long long)entry.st_dev, (unsigned long long)entry.st_ino);
    int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (fd < 0) return -1;
    auto decline = [&]() { close(fd); return -1; };
    if (connect(fd, (sockaddr *)&address, offsetof(sockaddr_un, sun_path) + count + 1) < 0) return decline();
    ucred peer{};
    socklen_t size = sizeof(peer);
    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &peer, &size) || peer.uid != geteuid()) return decline();
    // Only a compatible Python interpreter can receive the caller's streams
    // and environment. The server also authenticates this supervisor.
    std::string proc = "/proc/" + std::to_string(peer.pid);
    std::ifstream command(proc + "/cmdline", std::ios::binary);
    std::string executable, flag, module;
    if (!std::getline(command, executable, '\0') || !std::getline(command, flag, '\0') ||
        !std::getline(command, module, '\0') || flag != "-m" || module != "engine.fork_server")
        return decline();
    command.close();
    char *path = getenv("PATH");
    if (!path) return decline();
    struct stat python{}, serving{};
    if (stat((proc + "/exe").c_str(), &serving)) return decline();
    std::string paths(path);
    size_t pos = 0;
    for (;;) {
        size_t end = paths.find(':', pos);
        std::string dir = paths.substr(pos, end - pos);
        std::string exe = (dir.empty() ? "." : dir) + "/python3";
        if (!access(exe.c_str(), X_OK) && !stat(exe.c_str(), &python) && S_ISREG(python.st_mode)) break;
        if (end == std::string::npos) return decline();
        pos = end + 1;
    }
    if (python.st_dev != serving.st_dev || python.st_ino != serving.st_ino ||
        !only_cli_descriptors(control[0], control[1])) return decline();

    uint32_t argc = 0;
    while (args[argc + 1]) ++argc;
    std::string packet(reinterpret_cast<char *>(&argc), sizeof(argc));
    auto append = [&](const char *s) {
        size_t n = strlen(s) + 1;
        if (n > 65536 - packet.size()) return false;
        packet.append(s, n);
        return true;
    };
    for (char **arg = args + 1; *arg; ++arg) if (!append(*arg)) return decline();
    // The server reads the original environment through /proc under the
    // kernel's access rules. Only these two variables changed after exec.
    for (auto key : {"ATLAS_PROGRESS_FILE", "ATLAS_PROGRESS_OWNED"}) {
        const char *value = getenv(key);
        if (!append(value ? value : "")) return decline();
    }
    int cwd = open(".", O_DIRECTORY | O_RDONLY | O_CLOEXEC);
    if (cwd < 0) return decline();
    int passed[] = {0, 1, 2, control[1], cwd};
    iovec vec{packet.data(), packet.size()};
    alignas(cmsghdr) char ancillary[CMSG_SPACE(sizeof(passed))]{};
    msghdr message{};
    message.msg_iov = &vec;
    message.msg_iovlen = 1;
    message.msg_control = ancillary;
    message.msg_controllen = sizeof(ancillary);
    auto *header = CMSG_FIRSTHDR(&message);
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_RIGHTS;
    header->cmsg_len = CMSG_LEN(sizeof(passed));
    memcpy(CMSG_DATA(header), passed, sizeof(passed));
    auto written = sendmsg(fd, &message, MSG_NOSIGNAL);
    close(cwd);
    pollfd ready{fd, POLLIN, 0};
    char response;
    if (written != static_cast<ssize_t>(packet.size()) || poll(&ready, 1, 30) != 1 ||
        recv(fd, &response, 1, 0) != 1 || response != 'R' || send(fd, "G", 1, MSG_NOSIGNAL) != 1)
        return decline();
    return fd;
}
}
