#include <iostream>
#include <filesystem>
#include <stack>
#include <iterator>

class RecursiveDirIterator {
    private:
        void advance() {
            while (!dir_iter_stack_.empty()) {
                std::filesystem::directory_iterator& iter = dir_iter_stack_.top();

                if (iter == std::filesystem::end(iter)) {
                    dir_iter_stack_.pop();
                    continue;
                } else {
                    const std::filesystem::path& path = *iter;
                    if (std::filesystem::is_directory(path)) {
                        dir_iter_stack_.push(std::filesystem::directory_iterator(path));
                        ++iter;
                    } else {
                        current_file_ = path;
                        ++iter;
                        return;
                    }
                }
            }

            current_file_ = std::filesystem::path();
        }

        std::stack<std::filesystem::directory_iterator> dir_iter_stack_;
        std::filesystem::path current_file_;

    public:
        RecursiveDirIterator() = default;

        RecursiveDirIterator(const std::filesystem::path& path) : dir_iter_stack_() {
            if (!std::filesystem::exists(path)) {
                throw std::invalid_argument("Path does not exist: " + path.string());
            }
            if (!std::filesystem::is_directory(path)) { 
                throw std::invalid_argument("Path is not a directory: " + path.string());
            }
            dir_iter_stack_.push(std::filesystem::directory_iterator(path));
            advance();
        }

        const std::filesystem::path& operator*() const {
            return current_file_;
        }

        RecursiveDirIterator& operator++() {
            advance();
            return *this;
        }

        bool operator==(const RecursiveDirIterator& other) const {
            return current_file_ == other.current_file_ && dir_iter_stack_.empty() && other.dir_iter_stack_.empty();
        }

        bool operator!=(const RecursiveDirIterator& other) const {
            return !(*this == other);
        }
};

class RecursiveDirIterable {
    public:
        RecursiveDirIterable() = default;

        RecursiveDirIterable(const std::filesystem::path& root_path) : root_path_(root_path) {}

        RecursiveDirIterator begin() const {
            return RecursiveDirIterator(root_path_);
        }

        RecursiveDirIterator end() const {
            return RecursiveDirIterator();
        }

    private:
        std::filesystem::path root_path_;
};

