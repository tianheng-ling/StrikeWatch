# parsing input parameters
set report_dir [lindex $argv 0]
set fpga_type [lindex $argv 1] 
puts "$report_dir"
puts "FPGA Part: $fpga_type"

# create project
set proj_name "proj_resource"
create_project -force $proj_name ./$proj_name

# set the target device
# set_property part xc7s15ftgb196-2 [current_project]
set_property part $fpga_type [current_project]

# adding VHDL source files
add_files -scan_for_includes {./source}
update_compile_order -fileset sources_1
set_property default_lib work [current_project]
set_property library work [get_files -of [get_filesets {sources_1}]]
update_compile_order -fileset sources_1

# wait 5 s to load the design
after 5000

puts "Checking files in project:"
get_files -of [get_filesets sources_1]

# set the top module
set_property top env5_top_reconfig [current_fileset]
update_compile_order -fileset sources_1

# add resource constraints
add_files -fileset constrs_1 -norecurse {./constraints/env5_resource.xdc}

# run FPGA synthesis with error catching
puts "Starting synthesis..."

if {[catch {
    reset_run synth_1
    launch_runs synth_1 -jobs 8
    wait_on_run synth_1
} err]} {
    puts "ERROR: Synthesis failed or resource usage exceeded"
    puts "Vivado Error: $err"
    exit 1 # exit right now if synthesis fails
}

# try opening run (could still fail if previous step silent-failed)
if {[catch {open_run synth_1 -name synth_1} err]} {
    puts "ERROR: Failed to open synthesis results."
    puts "Vivado Error: $err"
    exit 1
}

# generate resource utilization reports
report_utilization -file "$report_dir/utilization_report.txt"
report_utilization -hierarchical -file "$report_dir/utilization_breakdown.txt"
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins -routable_nets -name timing_1 -file "$report_dir/timing_report.txt"

exit 0 # exit successfully if everything went well
