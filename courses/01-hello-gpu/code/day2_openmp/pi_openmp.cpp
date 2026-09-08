#include <cmath>
#include <iomanip>
#include <iostream>
#include <omp.h>

static double pi_serial(long long n) {
    const double step = 1.0 / static_cast<double>(n);
    double sum = 0.0;
    for (long long i = 0; i < n; ++i) {
        const double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    return sum * step;
}

static double pi_parallel(long long n) {
    const double step = 1.0 / static_cast<double>(n);
    double sum = 0.0;

    #pragma omp parallel for default(none) shared(n, step) reduction(+:sum)
    for (long long i = 0; i < n; ++i) {
        const double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    return sum * step;
}

int main(int argc, char** argv) {
    const long long n = argc > 1 ? std::stoll(argv[1]) : 500'000'000LL;
    const double reference = 3.14159265358979323846;

    const double s0 = omp_get_wtime();
    const double serial_value = pi_serial(n);
    const double serial_seconds = omp_get_wtime() - s0;

    const double p0 = omp_get_wtime();
    const double parallel_value = pi_parallel(n);
    const double parallel_seconds = omp_get_wtime() - p0;

    std::cout << std::setprecision(17)
              << "threads        = " << omp_get_max_threads() << '\n'
              << "n              = " << n << '\n'
              << "serial pi      = " << serial_value << '\n'
              << "parallel pi    = " << parallel_value << '\n'
              << "serial time    = " << serial_seconds << " s\n"
              << "parallel time  = " << parallel_seconds << " s\n"
              << "speedup        = " << serial_seconds / parallel_seconds << "x\n"
              << "serial error   = " << std::abs(serial_value - reference) << '\n'
              << "parallel error = " << std::abs(parallel_value - reference) << '\n';
}
