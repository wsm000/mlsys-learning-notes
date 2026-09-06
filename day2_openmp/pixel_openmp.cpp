#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

struct Pixel {
    std::uint8_t r, g, b;
};

static bool operator==(const Pixel& a, const Pixel& b) {
    return a.r == b.r && a.g == b.g && a.b == b.b;
}

static_assert(sizeof(Pixel) == 3, "Pixel must contain exactly RGB bytes");

struct Image {
    int width = 0;
    int height = 0;
    std::vector<Pixel> pixels;
};

static Image read_ppm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open input: " + path);

    std::string magic;
    int max_value = 0;
    in >> magic >> std::ws;
    if (magic != "P6") throw std::runtime_error("only binary P6 PPM is supported");

    // This teaching version expects a simple P6 header without comments.
    Image image;
    in >> image.width >> image.height >> max_value;
    in.get(); // consume the single whitespace byte before RGB data
    if (!in || image.width <= 0 || image.height <= 0 || max_value != 255) {
        throw std::runtime_error("invalid PPM header");
    }

    image.pixels.resize(static_cast<std::size_t>(image.width) * image.height);
    in.read(reinterpret_cast<char*>(image.pixels.data()),
            static_cast<std::streamsize>(image.pixels.size() * sizeof(Pixel)));
    if (!in) throw std::runtime_error("truncated PPM pixel data");
    return image;
}

static void write_ppm(const std::string& path, const Image& image) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open output: " + path);
    out << "P6\n" << image.width << ' ' << image.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(image.pixels.data()),
              static_cast<std::streamsize>(image.pixels.size() * sizeof(Pixel)));
}

static Image pixelate_serial(const Image& input, int block_size) {
    Image output{input.width, input.height, input.pixels};
    const int blocks_x = (input.width + block_size - 1) / block_size;
    const int blocks_y = (input.height + block_size - 1) / block_size;

    for (int k = 0; k < blocks_x * blocks_y; ++k) {
        const int block_x = k % blocks_x;
        const int block_y = k / blocks_x;
        const int x0 = block_x * block_size;
        const int y0 = block_y * block_size;
        const int x1 = std::min(x0 + block_size, input.width);
        const int y1 = std::min(y0 + block_size, input.height);

        long long r = 0, g = 0, b = 0;
        long long count = 0;
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                const Pixel p = input.pixels[static_cast<std::size_t>(y) * input.width + x];
                r += p.r; g += p.g; b += p.b; ++count;
            }
        }
        const Pixel average{
            static_cast<std::uint8_t>(r / count),
            static_cast<std::uint8_t>(g / count),
            static_cast<std::uint8_t>(b / count)};

        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                output.pixels[static_cast<std::size_t>(y) * input.width + x] = average;
            }
        }
    }
    return output;
}

static Image pixelate_parallel(const Image& input, int block_size) {
    Image output{input.width, input.height, input.pixels};
    const int blocks_x = (input.width + block_size - 1) / block_size;
    const int blocks_y = (input.height + block_size - 1) / block_size;
    const int total_blocks = blocks_x * blocks_y;

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < total_blocks; ++k) {
        const int block_x = k % blocks_x;
        const int block_y = k / blocks_x;
        const int x0 = block_x * block_size;
        const int y0 = block_y * block_size;
        const int x1 = std::min(x0 + block_size, input.width);
        const int y1 = std::min(y0 + block_size, input.height);

        // These accumulators are inside the loop: private to this iteration/thread.
        long long r = 0, g = 0, b = 0;
        long long count = 0;
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                const Pixel p = input.pixels[static_cast<std::size_t>(y) * input.width + x];
                r += p.r; g += p.g; b += p.b; ++count;
            }
        }
        const Pixel average{
            static_cast<std::uint8_t>(r / count),
            static_cast<std::uint8_t>(g / count),
            static_cast<std::uint8_t>(b / count)};

        // Each block owns a disjoint output rectangle: no write-write race.
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                output.pixels[static_cast<std::size_t>(y) * input.width + x] = average;
            }
        }
    }
    return output;
}

static bool same_image(const Image& a, const Image& b) {
    return a.width == b.width && a.height == b.height && a.pixels == b.pixels;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: pixel_openmp input.ppm output.ppm [block_size]\n";
        return 2;
    }
    const Image input = read_ppm(argv[1]);
    const std::string output_path = argv[2];
    const int block_size = argc >= 4 ? std::stoi(argv[3]) : 8;
    if (block_size <= 0) throw std::runtime_error("block_size must be positive");

    const double s0 = omp_get_wtime();
    const Image serial = pixelate_serial(input, block_size);
    const double serial_seconds = omp_get_wtime() - s0;

    const double p0 = omp_get_wtime();
    const Image parallel = pixelate_parallel(input, block_size);
    const double parallel_seconds = omp_get_wtime() - p0;

    write_ppm(output_path, parallel);
    std::cout << "image          = " << input.width << 'x' << input.height << '\n'
              << "block size     = " << block_size << '\n'
              << "max threads    = " << omp_get_max_threads() << '\n'
              << "serial time    = " << serial_seconds << " s\n"
              << "parallel time  = " << parallel_seconds << " s\n"
              << "speedup        = " << serial_seconds / parallel_seconds << "x\n"
              << "same output    = " << (same_image(serial, parallel) ? "YES" : "NO") << '\n'
              << "output         = " << output_path << '\n';
}
