# Clock Frequency
create_clock -name clk -period 50 [get_ports clk]

# SPI 1 MHz
# create_clock -name spi_clk -period 1000 [get_ports spi_clk]
