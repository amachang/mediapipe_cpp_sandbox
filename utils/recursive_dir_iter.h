#include <iostream>
#include <filesystem>
#include <stack>
#include <iterator>

class RecursiveDirIterator {
    private:
        void advance() {
            while (!dir_iter_stack.empty()) {
                std::filesystem::directory_iterator& iter = dir_iter_stack.top();

                if (iter == std::filesystem::end(iter)) {
                    dir_iter_stack.pop();
                    continue;
                } else {
                    const std::filesystem::path& path = *iter;
                    if (std::filesystem::is_directory(path)) {
                        dir_iter_stack.push(std::filesystem::directory_iterator(path));
                        ++iter;
                    } else {
                        current_file = path;
                        ++iter;
                        return;
                    }
                }
            }

            current_file = std::filesystem::path();
        }

        std::stack<std::filesystem::directory_iterator> dir_iter_stack;
        std::filesystem::path current_file;

    public:
        RecursiveDirIterator() = default;

        RecursiveDirIterator(const std::filesystem::path& path) : dir_iter_stack() {
            if (!std::filesystem::exists(path)) {
                throw std::invalid_argument("Path does not exist: " + path.string());
            }
            if (!std::filesystem::is_directory(path)) { 
                throw std::invalid_argument("Path is not a directory: " + path.string());
            }
            dir_iter_stack.push(std::filesystem::directory_iterator(path));
            advance();
        }

        const std::filesystem::path& operator*() const {
            return current_file;
        }

        RecursiveDirIterator& operator++() {
            advance();
            return *this;
        }

        bool operator==(const RecursiveDirIterator& other) const {
            return current_file == other.current_file && dir_iter_stack.empty() && other.dir_iter_stack.empty();
        }

        bool operator!=(const RecursiveDirIterator& other) const {
            return !(*this == other);
        }
};

class RecursiveDirIterable {
    public:
        RecursiveDirIterable(const std::filesystem::path& path) : root_path(path) {}

        RecursiveDirIterator begin() const {
            return RecursiveDirIterator(root_path);
        }

        RecursiveDirIterator end() const {
            return RecursiveDirIterator();
        }

    private:
        std::filesystem::path root_path;
};

