// =============================================================
//  WEATHER DATA ANALYZER  -  DAA Mini Project
//
//  What this program does:
//    1. Reads a weather CSV file given by the user
//    2. Finds the mean, min, max of every column
//    3. Finds the COVARIANCE matrix using matrix multiplication
//       done in 3 ways:  Sequential, Row-wise threads, Column-wise threads
//    4. Compares the time taken by the 3 ways
//    5. Finds the CORRELATION matrix and a simple prediction formula
//    6. Saves all results as CSV files (used by the Streamlit app)
//
//  Compile : g++ -O2 -std=c++17 -pthread weather_analyzer.cpp -o weather_analyzer
//  Run     : ./weather_analyzer sample_weather.csv results
// =============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <filesystem>

using namespace std;

// A matrix is just a 2D vector of numbers
typedef vector<vector<double>> Matrix;


// -------------------------------------------------------------
// PART 1 : READING THE CSV FILE
// -------------------------------------------------------------

// Checks if a piece of text is a number. If yes, stores it in 'value'.
bool isNumber(string text, double &value) {
    if (text.empty()) return false;
    try {
        size_t used;
        value = stod(text, &used);
        return used == text.size();   // whole text must be a number
    } catch (...) {
        return false;
    }
}

// Splits one line "22.5,65,1013" into {"22.5","65","1013"}
vector<string> splitLine(string line) {
    vector<string> cells;
    string cell;
    stringstream ss(line);
    while (getline(ss, cell, ',')) {
        if (!cell.empty() && cell.back() == '\r') cell.pop_back();  // Windows line ending
        cells.push_back(cell);
    }
    return cells;
}

// Reads the CSV file.
// - First line = column names
// - Columns with text (like Date or City) are ignored
// - Rows with missing values are skipped
bool readCSV(string fileName, Matrix &data, vector<string> &columnNames, int &skippedRows) {
    ifstream file(fileName);
    if (!file.is_open()) {
        cout << "Error: cannot open file " << fileName << endl;
        return false;
    }

    string line;
    getline(file, line);
    vector<string> allNames = splitLine(line);

    // Read all remaining lines as text first
    vector<vector<string>> rows;
    while (getline(file, line)) {
        if (!line.empty()) rows.push_back(splitLine(line));
    }
    if (rows.empty()) {
        cout << "Error: file has no data rows" << endl;
        return false;
    }

    // Decide which columns are numeric by looking at the first data row
    vector<int> numericColumns;
    for (int c = 0; c < (int)allNames.size(); c++) {
        double v;
        if (c < (int)rows[0].size() && isNumber(rows[0][c], v)) {
            numericColumns.push_back(c);
            columnNames.push_back(allNames[c]);
        }
    }
    if (numericColumns.size() < 2) {
        cout << "Error: need at least 2 numeric columns" << endl;
        return false;
    }

    // Convert each row to numbers; skip the row if any value is bad
    skippedRows = 0;
    for (auto &row : rows) {
        vector<double> numbers;
        bool good = true;
        for (int c : numericColumns) {
            double v;
            if (c < (int)row.size() && isNumber(row[c], v)) numbers.push_back(v);
            else { good = false; break; }
        }
        if (good) data.push_back(numbers);
        else skippedRows++;
    }
    return data.size() >= 2;
}


// -------------------------------------------------------------
// PART 2 : MATRIX MULTIPLICATION  (C = A x B)
// -------------------------------------------------------------

// Method 1: Sequential (one thread does everything)
// Time complexity: O(rows of A * cols of B * cols of A)
void multiplySequential(const Matrix &A, const Matrix &B, Matrix &C) {
    for (int i = 0; i < (int)A.size(); i++) {
        for (int j = 0; j < (int)B[0].size(); j++) {
            double sum = 0;
            for (int k = 0; k < (int)B.size(); k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
}

// Work done by ONE thread in row-wise method: rows from startRow to endRow-1
void rowWork(const Matrix &A, const Matrix &B, Matrix &C, int startRow, int endRow) {
    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < (int)B[0].size(); j++) {
            double sum = 0;
            for (int k = 0; k < (int)B.size(); k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
}

// Method 2: Row-wise threading (rows of C are divided among threads)
void multiplyRowThreads(const Matrix &A, const Matrix &B, Matrix &C, int numThreads) {
    int totalRows = A.size();
    if (numThreads > totalRows) numThreads = totalRows;
    int rowsPerThread = totalRows / numThreads;

    vector<thread> threads;
    for (int t = 0; t < numThreads; t++) {
        int start = t * rowsPerThread;
        int end = (t == numThreads - 1) ? totalRows : start + rowsPerThread;  // last thread takes leftovers
        threads.push_back(thread(rowWork, cref(A), cref(B), ref(C), start, end));
    }
    for (auto &th : threads) th.join();   // wait for all threads to finish
}

// Work done by ONE thread in column-wise method: columns from startCol to endCol-1
void colWork(const Matrix &A, const Matrix &B, Matrix &C, int startCol, int endCol) {
    for (int i = 0; i < (int)A.size(); i++) {
        for (int j = startCol; j < endCol; j++) {
            double sum = 0;
            for (int k = 0; k < (int)B.size(); k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
}

// Method 3: Column-wise threading (columns of C are divided among threads)
void multiplyColThreads(const Matrix &A, const Matrix &B, Matrix &C, int numThreads) {
    int totalCols = B[0].size();
    if (numThreads > totalCols) numThreads = totalCols;
    int colsPerThread = totalCols / numThreads;

    vector<thread> threads;
    for (int t = 0; t < numThreads; t++) {
        int start = t * colsPerThread;
        int end = (t == numThreads - 1) ? totalCols : start + colsPerThread;
        threads.push_back(thread(colWork, cref(A), cref(B), ref(C), start, end));
    }
    for (auto &th : threads) th.join();
}

// Returns true if two matrices have the same values
bool sameMatrix(const Matrix &X, const Matrix &Y) {
    for (int i = 0; i < (int)X.size(); i++)
        for (int j = 0; j < (int)X[0].size(); j++)
            if (fabs(X[i][j] - Y[i][j]) > 1e-6) return false;
    return true;
}

// Current time in milliseconds (used to measure how long something takes)
double nowMs() {
    return chrono::duration<double, milli>(
        chrono::high_resolution_clock::now().time_since_epoch()).count();
}


// -------------------------------------------------------------
// PART 3 : SAVING RESULTS
// -------------------------------------------------------------

// Saves a square matrix with column names as headers
void saveMatrix(string fileName, const Matrix &M, const vector<string> &names) {
    ofstream f(fileName);
    f << "Feature";
    for (auto &n : names) f << "," << n;
    f << "\n";
    for (int i = 0; i < (int)M.size(); i++) {
        f << names[i];
        for (double v : M[i]) f << "," << v;
        f << "\n";
    }
}

// Prints a square matrix neatly on the screen
void printMatrix(string title, const Matrix &M, const vector<string> &names) {
    cout << "\n" << title << "\n";
    cout << setw(16) << "";
    for (auto &n : names) cout << setw(15) << n.substr(0, 14);
    cout << "\n";
    for (int i = 0; i < (int)M.size(); i++) {
        cout << setw(16) << names[i].substr(0, 15);
        for (double v : M[i]) cout << setw(15) << fixed << setprecision(3) << v;
        cout << "\n";
    }
}


// -------------------------------------------------------------
// MAIN PROGRAM
// -------------------------------------------------------------
int main(int argc, char *argv[]) {

    // ---- STEP 1: Get the input file name ----
    string inputFile, outputFolder = "results";
    if (argc >= 2) inputFile = argv[1];
    else {
        cout << "Enter weather CSV file name: ";
        cin >> inputFile;
    }
    if (argc >= 3) outputFolder = argv[2];
    filesystem::create_directories(outputFolder);

    // ---- STEP 2: Read the data ----
    Matrix data;                 // n rows x m columns
    vector<string> names;        // column names
    int skipped = 0;
    if (!readCSV(inputFile, data, names, skipped)) return 1;

    int n = data.size();         // number of records (rows)
    int m = names.size();        // number of weather factors (columns)

    cout << "==============================================\n";
    cout << "          WEATHER DATA ANALYZER\n";
    cout << "==============================================\n";
    cout << "File          : " << inputFile << "\n";
    cout << "Records (n)   : " << n << "\n";
    cout << "Factors (m)   : " << m << "\n";
    cout << "Rows skipped  : " << skipped << "\n";

    // ---- STEP 3: Mean, Min, Max of every column ----
    vector<double> mean(m, 0), minV(m), maxV(m), stdDev(m);
    for (int j = 0; j < m; j++) {
        minV[j] = maxV[j] = data[0][j];
        for (int i = 0; i < n; i++) {
            mean[j] += data[i][j];
            minV[j] = min(minV[j], data[i][j]);
            maxV[j] = max(maxV[j], data[i][j]);
        }
        mean[j] = mean[j] / n;
    }

    // ---- STEP 4: Centered data  Xc = X - mean  and its transpose XcT ----
    Matrix Xc(n, vector<double>(m));
    Matrix XcT(m, vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            Xc[i][j] = data[i][j] - mean[j];
            XcT[j][i] = Xc[i][j];
        }
    }

    // ---- STEP 5: XcT x Xc using the 3 methods and measure time ----
    // Each method is repeated many times and the average is taken,
    // because one run is too fast to measure accurately.
    int numThreads = thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    int repeat = 10;

    Matrix C1(m, vector<double>(m)), C2(m, vector<double>(m)), C3(m, vector<double>(m));
    double start;

    start = nowMs();
    for (int r = 0; r < repeat; r++) multiplySequential(XcT, Xc, C1);
    double timeSeq = (nowMs() - start) / repeat;

    start = nowMs();
    for (int r = 0; r < repeat; r++) multiplyRowThreads(XcT, Xc, C2, numThreads);
    double timeRow = (nowMs() - start) / repeat;

    start = nowMs();
    for (int r = 0; r < repeat; r++) multiplyColThreads(XcT, Xc, C3, numThreads);
    double timeCol = (nowMs() - start) / repeat;

    bool allSame = sameMatrix(C1, C2) && sameMatrix(C1, C3);

    // ---- STEP 6: Covariance = (XcT x Xc) / (n-1) ----
    Matrix cov(m, vector<double>(m));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            cov[i][j] = C1[i][j] / (n - 1);

    // Standard deviation = square root of variance (diagonal of covariance)
    for (int j = 0; j < m; j++) stdDev[j] = sqrt(cov[j][j]);

    // ---- STEP 7: Correlation = Cov(X,Y) / (stdDev X * stdDev Y)  -> between -1 and +1 ----
    Matrix corr(m, vector<double>(m));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            corr[i][j] = (stdDev[i] > 0 && stdDev[j] > 0) ? cov[i][j] / (stdDev[i] * stdDev[j]) : 0;

    // ---- STEP 8: Print results on screen ----
    cout << "\nSTATISTICS\n";
    cout << setw(16) << "Factor" << setw(12) << "Mean" << setw(12) << "StdDev"
         << setw(12) << "Min" << setw(12) << "Max" << "\n";
    for (int j = 0; j < m; j++) {
        cout << setw(16) << names[j].substr(0, 15) << fixed << setprecision(2)
             << setw(12) << mean[j] << setw(12) << stdDev[j]
             << setw(12) << minV[j] << setw(12) << maxV[j] << "\n";
    }

    printMatrix("COVARIANCE MATRIX", cov, names);
    printMatrix("CORRELATION MATRIX  (+1 = rise together, -1 = opposite, 0 = no link)", corr, names);

    cout << "\nTIME COMPARISON  (threads used = " << numThreads << ")\n";
    cout << fixed << setprecision(4);
    cout << "  Sequential            : " << timeSeq << " ms\n";
    cout << "  Row-wise threading    : " << timeRow << " ms   (speedup " << setprecision(2) << timeSeq / timeRow << "x)\n";
    cout << setprecision(4);
    cout << "  Column-wise threading : " << timeCol << " ms   (speedup " << setprecision(2) << timeSeq / timeCol << "x)\n";
    cout << "  All 3 methods give same answer : " << (allSame ? "YES" : "NO") << "\n";

    // ---- STEP 9: Save results as CSV files for the Streamlit app ----
    cout << setprecision(6);
    saveMatrix(outputFolder + "/covariance.csv", cov, names);
    saveMatrix(outputFolder + "/correlation.csv", corr, names);

    ofstream statsFile(outputFolder + "/statistics.csv");
    statsFile << "Feature,Mean,StdDev,Min,Max\n";
    for (int j = 0; j < m; j++)
        statsFile << names[j] << "," << mean[j] << "," << stdDev[j] << "," << minV[j] << "," << maxV[j] << "\n";

    // Prediction formula for every pair:  Y = slope * X + intercept
    // slope = Cov(X,Y) / Var(X)
    ofstream regFile(outputFolder + "/regression.csv");
    regFile << "X,Y,Slope,Intercept,Correlation\n";
    for (int x = 0; x < m; x++) {
        for (int y = 0; y < m; y++) {
            if (x == y || cov[x][x] == 0) continue;
            double slope = cov[x][y] / cov[x][x];
            double intercept = mean[y] - slope * mean[x];
            regFile << names[x] << "," << names[y] << "," << slope << "," << intercept << "," << corr[x][y] << "\n";
        }
    }

    ofstream timeFile(outputFolder + "/timing.csv");
    timeFile << "Method,Threads,Time_ms\n";
    timeFile << "Sequential,1," << timeSeq << "\n";
    timeFile << "Row_Wise_Threading," << numThreads << "," << timeRow << "\n";
    timeFile << "Column_Wise_Threading," << numThreads << "," << timeCol << "\n";

    ofstream infoFile(outputFolder + "/info.csv");
    infoFile << "Rows,Columns,SkippedRows,CPUThreads,ResultsMatch\n";
    infoFile << n << "," << m << "," << skipped << "," << numThreads << "," << (allSame ? "Yes" : "No") << "\n";

    cout << "\nResults saved in folder: " << outputFolder << "/\n";
    return 0;
}
