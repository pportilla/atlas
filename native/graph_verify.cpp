// The isolated graph verifier program. See graph_verify_entry.cpp.
extern "C" int atlas_graph_verify_main(int, char**) noexcept;

int main(int argc, char** argv) {
    return atlas_graph_verify_main(argc, argv);
}
