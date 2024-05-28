#include <iostream>
#include <inja/inja.hpp>

int main() {
    inja::Environment env;
    inja::Template temp = env.parse("Hello, {{ name }}!");
    std::string result = env.render(temp, {{"name", "world"}});
    std::cout << result << std::endl;
    return 0;
}

