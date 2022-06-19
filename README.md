# parallel-printer-symmetry

This project attempts to employ a symmetry-based algorithm to achieve optimal usage of multiple homogeneous printers for printing any input model.

**Parallel Printer/**: Directory containing the C++ project that implements the algorithm. CMake 3.15+ required, as well as CGAL 4.x (CGAL 5.x fails to compile due to removal of exporting utilities). The project was compiled using G++ with C++17.

**Models/**: Directory containing the models selected/sampled for testing.

**Decomposed Models/**: Directory containing the output models used in the quantitative phase of the research.

**Data Processing.ods**: LibreOffice Calc spreadsheet containing the data analysis used.

