#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

#include "mpi.h"

const int LOOP_WAIT = 1;
const size_t NUM_COLUMNS = 9;
const std::vector<std::string> COLUMN_NAMES = {
    "x", "y", "z", "avg_af", "avg_relV", "H_index", "var_af", "drift_flux", "avg_dPdy"
};
const size_t BLOCK_SIZE = 16;
const size_t NUM_ELEMENTS = NUM_COLUMNS*BLOCK_SIZE;

std::vector<int> IDS = {19909, 20450, 20966, 21480};

void printLines(const std::vector<std::vector<float>>& data) {

    for (size_t row = 0; row < BLOCK_SIZE; row++) {
        for (size_t col = 0; col<NUM_COLUMNS; col++) {
            std::cout << data[col][row] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "---" << std::endl;
}

int main() {

    MPI_Init(NULL, NULL);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string fname = std::format("fml_plt{}_lev0_fs0032.csv", IDS[rank]);
    std::ifstream file(fname);
    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }

    std::string header;
    std::getline(file, header); // Read the header row

    std::string line;
    int lineCount = 0;
    int num_blocks = 0;
    std::vector<std::vector<float>> data(NUM_COLUMNS);
    for (size_t col=0; col<NUM_COLUMNS; col++) data[col].reserve(BLOCK_SIZE);

    while (std::getline(file, line)) {

        std::stringstream ss(line);
        std::string column;
        std::vector<float> row;
        size_t col = 0;

        while (std::getline(ss, column, ',') && row.size() < NUM_COLUMNS) {
            data[col].push_back(std::stof(column));
            col++;
        }

        lineCount++;

        if (lineCount == BLOCK_SIZE) {
            num_blocks++;
            std::cout << num_blocks << std::endl;
            printLines(data);
            for (size_t i=0; i<NUM_COLUMNS; i++) data[i].clear();
            lineCount = 0;
            std::this_thread::sleep_for(std::chrono::seconds(LOOP_WAIT));
        }
    }

    file.close();
}