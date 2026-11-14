// Target-independent Triton IR for finite-field S-pair reduction.
// Each program owns one row. The host supplies a topological ordering of
// its triangular reducer graph, split into levels of independent columns.
// Each 16-column tile sums incoming products in four lanes per column.
// Reducing modulo p^2 after each product keeps signed 64-bit sums exact.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace atlas {
std::string triton_trace_ir(const std::string& name, const std::vector<int>& parameters) {
    if (name == "trace_probe" && parameters.empty())
        return R"IR(module {
  tt.func public @trace_probe(%buffer: !tt.ptr<i64>) {
    %modulus = arith.constant 1000000007 : i64
    %value = tt.load %buffer : !tt.ptr<i64>
    %square = arith.muli %value, %value : i64
    %residue = arith.remsi %square, %modulus : i64
    tt.store %buffer, %residue : !tt.ptr<i64>
    tt.return
  }
})IR";
    if (name != "trace_levels" || parameters.size() != 1 || parameters.front() < 2)
        throw std::invalid_argument("Invalid trace reduction modulus.");
    const int64_t p = parameters.front();
    const auto prime = std::to_string(p), square = std::to_string(p * p);
    return R"IR(module {
  tt.func public @trace_levels(%initial: !tt.ptr<i32>, %pivots: !tt.ptr<i32>, %starts: !tt.ptr<i32>, %parents: !tt.ptr<i32>, %factors: !tt.ptr<i32>, %order: !tt.ptr<i32>, %levels: !tt.ptr<i32>, %work: !tt.ptr<i32>, %output: !tt.ptr<i32>, %width: i32, %depth: i32) {
    %c4_i32 = arith.constant 4 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %zero_residues = arith.constant dense<0> : tensor<16xi64>
    %lane_primes = arith.constant dense<)IR" + prime + R"IR(> : tensor<16xi64>
    %primes = arith.constant dense<)IR" + prime + R"IR(> : tensor<16x4xi64>
    %squared_primes = arith.constant dense<)IR" + square + R"IR(> : tensor<16x4xi64>
    %zero_edges = arith.constant dense<0> : tensor<16x4xi32>
    %zero_sums = arith.constant dense<0> : tensor<16x4xi64>
    %ones = arith.constant dense<1> : tensor<16xi32>
    %zero_columns = arith.constant dense<0> : tensor<16xi32>
    %c1_i32 = arith.constant 1 : i32
    %row = tt.get_program_id x : i32
    %row_7 = arith.muli %row, %width : i32
    %cols = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %terms = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    scf.for %d = %c0_i32 to %depth step %c1_i32 : i32 {
      %first = tt.addptr %levels, %d : !tt.ptr<i32>, i32
      %first_8 = tt.load %first : !tt.ptr<i32>
      %limit = tt.addptr %first, %c1_i32 : !tt.ptr<i32>, i32
      %limit_9 = tt.load %limit : !tt.ptr<i32>
      scf.for %base = %first_8 to %limit_9 step %c16_i32 : i32 {
        %pos = tt.splat %base : i32 -> tensor<16xi32>
        %pos_10 = arith.addi %pos, %cols : tensor<16xi32>
        %live = tt.splat %limit_9 : i32 -> tensor<16xi32>
        %live_11 = arith.cmpi slt, %pos_10, %live : tensor<16xi32>
        %col = tt.splat %order : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
        %col_12 = tt.addptr %col, %pos_10 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
        %col_13 = tt.load %col_12, %live_11, %zero_columns : tensor<16x!tt.ptr<i32>>
        %begin = tt.splat %starts : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
        %begin_14 = tt.addptr %begin, %col_13 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
        %begin_15 = tt.load %begin_14, %live_11, %zero_columns : tensor<16x!tt.ptr<i32>>
        %end = tt.addptr %begin_14, %ones : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
        %end_16 = tt.load %end, %live_11, %zero_columns : tensor<16x!tt.ptr<i32>>
        %longest = arith.subi %end_16, %begin_15 : tensor<16xi32>
        %longest_17 = "tt.reduce"(%longest) <{axis = 0 : i32}> ({
        ^bb0(%longest_28: i32, %longest_29: i32):
          %longest_30 = arith.maxsi %longest_28, %longest_29 : i32
          tt.reduce.return %longest_30 : i32
        }) : (tensor<16xi32>) -> i32
        %sums = scf.for %k = %c0_i32 to %longest_17 step %c4_i32 iter_args(%sums_28 = %zero_sums) -> (tensor<16x4xi64>) : i32 {
          %edge = tt.expand_dims %begin_15 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
          %edge_29 = tt.splat %k : i32 -> tensor<16x1xi32>
          %edge_30 = arith.addi %edge, %edge_29 : tensor<16x1xi32>
          %edge_31 = tt.expand_dims %terms {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
          %edge_32 = tt.broadcast %edge_30 : tensor<16x1xi32> -> tensor<16x4xi32>
          %edge_33 = tt.broadcast %edge_31 : tensor<1x4xi32> -> tensor<16x4xi32>
          %edge_34 = arith.addi %edge_32, %edge_33 : tensor<16x4xi32>
          %mask = tt.expand_dims %end_16 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
          %mask_35 = tt.broadcast %mask : tensor<16x1xi32> -> tensor<16x4xi32>
          %mask_36 = arith.cmpi slt, %edge_34, %mask_35 : tensor<16x4xi32>
          %src = tt.splat %parents : !tt.ptr<i32> -> tensor<16x4x!tt.ptr<i32>>
          %src_37 = tt.addptr %src, %edge_34 : tensor<16x4x!tt.ptr<i32>>, tensor<16x4xi32>
          %src_38 = tt.load %src_37, %mask_36, %zero_edges : tensor<16x4x!tt.ptr<i32>>
          %coeff = tt.splat %factors : !tt.ptr<i32> -> tensor<16x4x!tt.ptr<i32>>
          %coeff_39 = tt.addptr %coeff, %edge_34 : tensor<16x4x!tt.ptr<i32>>, tensor<16x4xi32>
          %coeff_40 = tt.load %coeff_39, %mask_36, %zero_edges : tensor<16x4x!tt.ptr<i32>>
          %coeff_41 = arith.extsi %coeff_40 : tensor<16x4xi32> to tensor<16x4xi64>
          %value = tt.addptr %work, %row_7 : !tt.ptr<i32>, i32
          %value_42 = tt.splat %value : !tt.ptr<i32> -> tensor<16x4x!tt.ptr<i32>>
          %value_43 = tt.addptr %value_42, %src_38 : tensor<16x4x!tt.ptr<i32>>, tensor<16x4xi32>
          %value_44 = tt.load %value_43, %mask_36, %zero_edges : tensor<16x4x!tt.ptr<i32>>
          %value_45 = arith.extsi %value_44 : tensor<16x4xi32> to tensor<16x4xi64>
          %sums_46 = arith.muli %coeff_41, %value_45 : tensor<16x4xi64>
          %sums_47 = arith.addi %sums_28, %sums_46 : tensor<16x4xi64>
          %sums_48 = arith.cmpi sge, %sums_47, %squared_primes : tensor<16x4xi64>
          %sums_49 = arith.subi %sums_47, %squared_primes : tensor<16x4xi64>
          %sums_50 = arith.select %sums_48, %sums_49, %sums_47 : tensor<16x4xi1>, tensor<16x4xi64>
          scf.yield %sums_50 : tensor<16x4xi64>
        }
        %result = arith.remsi %sums, %primes : tensor<16x4xi64>
        %result_18 = "tt.reduce"(%result) <{axis = 1 : i32}> ({
        ^bb0(%result_28: i64, %result_29: i64):
          %result_30 = arith.addi %result_28, %result_29 : i64
          tt.reduce.return %result_30 : i64
        }) : (tensor<16x4xi64>) -> tensor<16xi64>
        %result_19 = tt.addptr %initial, %row_7 : !tt.ptr<i32>, i32
        %result_20 = tt.splat %result_19 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
        %result_21 = tt.addptr %result_20, %col_13 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
        %result_22 = tt.load %result_21, %live_11, %zero_columns : tensor<16x!tt.ptr<i32>>
        %result_23 = arith.extsi %result_22 : tensor<16xi32> to tensor<16xi64>
        %result_24 = arith.addi %result_18, %result_23 : tensor<16xi64>
        %result_25 = arith.remsi %result_24, %lane_primes : tensor<16xi64>
        %pivot = tt.splat %pivots : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
        %pivot_26 = tt.addptr %pivot, %col_13 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
        %pivot_27 = tt.load %pivot_26, %live_11, %zero_columns : tensor<16x!tt.ptr<i32>>
        %0 = tt.addptr %work, %row_7 : !tt.ptr<i32>, i32
        %1 = tt.splat %0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
        %2 = tt.addptr %1, %col_13 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
        %3 = arith.subi %lane_primes, %result_25 : tensor<16xi64>
        %4 = arith.trunci %3 : tensor<16xi64> to tensor<16xi32>
        tt.store %2, %4, %live_11 : tensor<16x!tt.ptr<i32>>
        %5 = tt.addptr %output, %row_7 : !tt.ptr<i32>, i32
        %6 = tt.splat %5 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
        %7 = tt.addptr %6, %col_13 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
        %8 = arith.cmpi ne, %pivot_27, %zero_columns : tensor<16xi32>
        %9 = arith.select %8, %zero_residues, %result_25 : tensor<16xi1>, tensor<16xi64>
        %10 = arith.trunci %9 : tensor<16xi64> to tensor<16xi32>
        tt.store %7, %10, %live_11 : tensor<16x!tt.ptr<i32>>
      }
      ttg.barrier all
    }
    tt.return
  }
}
)IR";
}
} // namespace atlas
