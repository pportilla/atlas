// Isolated adapter for the same packed graph verifier as the general search.
// A nonzero exit leaves its normal, replay-aware Python worker in charge.
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>

extern "C" {
char* atlas_ideal(const char*, size_t) noexcept;
void atlas_groebner_release(char*) noexcept;
void atlas_groebner_activity_counts(uint64_t*) noexcept;
}

int main() {
    std::string input(std::istreambuf_iterator<char>(std::cin), {});
    if (input.size()>65536) return 1;
    std::unique_ptr<char, decltype(&atlas_groebner_release)> answer(nullptr,atlas_groebner_release);
    if (input.rfind("smooth_reduced_graph ",0) == 0) {
        answer.reset(atlas_ideal(input.data(), input.size()));
        if (!answer || !*answer.get()) return 1;
    } else {
        size_t start = 0;
        // Only publish candidate #3 when its predecessors both fail. Each frame
        // runs the same packed verification operation as exhaustive search.
        for (unsigned index=0; index<3; ++index) {
            auto stop = input.find('\f',start);
            if ((stop==std::string::npos)!=(index==2) || input.compare(start,7,"verify ")!=0) return 1;
            auto length = (stop==std::string::npos ? input.size() : stop)-start;
            answer.reset(atlas_ideal(input.data()+start,length));
            if (!answer || std::string_view(answer.get()).find(index==2 ? "pass " : "rejected ")!=0) return 1;
            start = stop==std::string::npos ? input.size() : stop+1;
        }
    }
    std::array<uint64_t,8> activity{};
    atlas_groebner_activity_counts(activity.data());
    // A traced/modular computation needs the request's replay hooks and
    // counters. Repeat it in the general worker before publishing any result.
    if (std::any_of(activity.begin(),activity.end(),[](uint64_t count){return count!=0;})) return 1;
    std::cout << answer.get();
    return std::cout ? 0 : 1;
}
