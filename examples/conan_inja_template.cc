#include <inja.hpp>

// Just for convenience
using namespace inja;

int main() {
    // Create a new Environment
    Environment env;

    // Render a template
    std::string result = env.render("Hello, {{ name }}!", {{"name", "world"}});

    // Print the result
    std::cout << result << std::endl;

    return 0;
}

