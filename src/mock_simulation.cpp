#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

#include "client.h"
#include "dataset.h"
#include "mpi.h"

const int LOOP_WAIT = 1;
const size_t NUM_COLUMNS = 9;
const size_t BLOCK_SIZE = 16;
const size_t NUM_ELEMENTS = NUM_COLUMNS*BLOCK_SIZE;

const std::string DS_LIST = "simulation_data";
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

void send_data(
    SmartRedis::Client& client,
    const std::vector<std::vector<float>>& data,
    const std::vector<std::string>& names,
    const size_t iteration,
    const std::string& rank_id
) {

    std::string ds_name = "data_iteration_" + std::to_string(iteration) + "_ID_" + rank_id;
    SmartRedis::DataSet ds(ds_name);

    for( size_t col=0; col<names.size(); col++) {
        ds.add_tensor(names[col], data[col].data(), {BLOCK_SIZE}, SRTensorTypeFloat, SRMemLayoutContiguous);
    }
    client.put_dataset(ds);
    client.append_to_list(DS_LIST, ds);

}
int main() {

    // Initialize MPI
    MPI_Init(NULL, NULL);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize SmartRedis
    std::string rank_id = std::to_string(IDS[rank]);
    SmartRedis::Client client(rank_id);

    // Setup file to read data from
    std::string fname = "data/fml_plt" + rank_id + "_lev0_fs0032.csv";
    std::ifstream file(fname);
    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }

    // Read the column nammes
    std::string line;
    std::string column;
    std::getline(file, line);
    std::stringstream ss_header(line);
    std::vector<std::string> col_names;
    while (std::getline(ss_header, column, ',')) col_names.push_back(column);

    // Prepare for loop
    int line_count = 0;
    int num_blocks = 0;
    std::vector<std::vector<float>> data(NUM_COLUMNS);
    for (size_t col=0; col<NUM_COLUMNS; col++) data[col].reserve(BLOCK_SIZE);
    size_t loop_index = 0;

    // Mimic the timestepping loop of the simulation
    // 1. Read from the CSV file with pre-computed data with 9 properties
    // 2. Every 10 lines (e.g. 10 timesteps)
    //   2a. Make a SmartSim dataset
    //   2b. Add a tensor for each of the properties
    //   2c. Send the dataset to the database
    //   2d. Add the dataset to a dataset list
    while (std::getline(file, line)) {

        loop_index++;
        line_count++;
        std::stringstream ss(line);
        std::vector<float> row;
        size_t col = 0;

        while (std::getline(ss, column, ',') && row.size() < NUM_COLUMNS) {
            data[col].push_back(std::stof(column));
            col++;
        }

        if (line_count == BLOCK_SIZE) {
            num_blocks++;

            for (auto & name : col_names) std::cout << name << " ";
            std::cout << std::endl;
            printLines(data);
            send_data(client, data, col_names, loop_index, rank_id);

            for (size_t i=0; i<NUM_COLUMNS; i++) data[i].clear();

            line_count = 0;
            std::this_thread::sleep_for(std::chrono::seconds(LOOP_WAIT));
        }
    }

    file.close();
    MPI_Finalize();
}