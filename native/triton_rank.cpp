// Target-independent GPU kernels for modular row witnesses, specialized to
// the block dimensions and the two prime fields in the Jacobian rank test.
#include <stdexcept>
#include <string>
#include <string_view>

namespace atlas {
namespace {

void substitute(std::string& text, std::string_view key, const std::string& value) {
    size_t position = 0;
    while ((position = text.find(key, position)) != std::string::npos) {
        text.replace(position, key.size(), value);
        position += value.size();
    }
}

constexpr std::string_view header = R"(module {
  tt.func public @$KERNEL(%matrix: !tt.ptr<i32>, %ids: !tt.ptr<i32>, %rows: i32, %cols: i32, %col: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %missing = arith.constant -1 : i32
    %state = tt.addptr %ids, %rows : !tt.ptr<i32>, i32
    %status = tt.load %state : !tt.ptr<i32>
)";

constexpr std::string_view search = R"(
    scf.if %enabled {
      %indices = tt.make_range {end = $R : i32, start = 0 : i32} : tensor<$Rxi32>
      %lo = tt.splat %target : i32 -> tensor<$Rxi32>
      %hi = tt.splat %rows : i32 -> tensor<$Rxi32>
      %after = arith.cmpi sge, %indices, %lo : tensor<$Rxi32>
      %before = arith.cmpi slt, %indices, %hi : tensor<$Rxi32>
      %active = arith.andi %after, %before : tensor<$Rxi1>
      %width = tt.splat %cols : i32 -> tensor<$Rxi32>
      %column = tt.addptr %matrix, %col : !tt.ptr<i32>, i32
      %columns = tt.splat %column : !tt.ptr<i32> -> tensor<$Rx!tt.ptr<i32>>
      %row_offsets = arith.muli %indices, %width : tensor<$Rxi32>
      %candidates = tt.addptr %columns, %row_offsets : tensor<$Rx!tt.ptr<i32>>, tensor<$Rxi32>
      %zeros = arith.constant dense<0> : tensor<$Rxi32>
      %values = tt.load %candidates, %active, %zeros : tensor<$Rx!tt.ptr<i32>>
      %nonzero = arith.cmpi ne, %values, %zeros : tensor<$Rxi32>
      %possible = arith.select %nonzero, %indices, %hi : tensor<$Rxi1>, tensor<$Rxi32>
      %selected = "tt.reduce"(%possible) <{axis = 0 : i32}> ({
      ^bb0(%a: i32, %b: i32):
        %m = arith.minsi %a, %b : i32
        tt.reduce.return %m : i32
      }) : (tensor<$Rxi32>) -> i32
      %absent = arith.cmpi eq, %selected, %rows : i32
      scf.if %absent {
        tt.store %flag, %missing : !tt.ptr<i32>
      } else {
        %offset = arith.muli %selected, %cols : i32
        %chosen = tt.addptr %matrix, %offset : !tt.ptr<i32>, i32
        %pivotptr = tt.addptr %chosen, %col : !tt.ptr<i32>, i32
        %pivotval = tt.load %pivotptr : !tt.ptr<i32>
        %base = arith.extui %pivotval : i32 to i64
        %prime = arith.constant $P : i64
        %unit = arith.constant 1 : i64
)";

constexpr std::string_view normalize = R"(
        %j = tt.make_range {end = $C : i32, start = 0 : i32} : tensor<$Cxi32>
        %first = tt.splat %col : i32 -> tensor<$Cxi32>
        %last = tt.splat %cols : i32 -> tensor<$Cxi32>
        %above = arith.cmpi sge, %j, %first : tensor<$Cxi32>
        %below = arith.cmpi slt, %j, %last : tensor<$Cxi32>
        %live = arith.andi %above, %below : tensor<$Cxi1>
        %bases = tt.splat %chosen : !tt.ptr<i32> -> tensor<$Cx!tt.ptr<i32>>
        %chosen_ptrs = tt.addptr %bases, %j : tensor<$Cx!tt.ptr<i32>>, tensor<$Cxi32>
        %col_zeros = arith.constant dense<0> : tensor<$Cxi32>
        %tail = tt.load %chosen_ptrs, %live, %col_zeros : tensor<$Cx!tt.ptr<i32>>
        %at = arith.muli %target, %cols : i32
        %dest = tt.addptr %matrix, %at : !tt.ptr<i32>, i32
        %dests = tt.splat %dest : !tt.ptr<i32> -> tensor<$Cx!tt.ptr<i32>>
        %dest_ptrs = tt.addptr %dests, %j : tensor<$Cx!tt.ptr<i32>>, tensor<$Cxi32>
        %swap = arith.cmpi ne, %selected, %target : i32
        scf.if %swap {
          %old = tt.load %dest_ptrs, %live, %col_zeros : tensor<$Cx!tt.ptr<i32>>
          tt.store %chosen_ptrs, %old, %live : tensor<$Cx!tt.ptr<i32>>
          %from = tt.addptr %ids, %selected : !tt.ptr<i32>, i32
          %to = tt.addptr %ids, %target : !tt.ptr<i32>, i32
          %old_id = tt.load %to : !tt.ptr<i32>
          %new_id = tt.load %from : !tt.ptr<i32>
          tt.store %from, %old_id : !tt.ptr<i32>
          tt.store %to, %new_id : !tt.ptr<i32>
        }
        %wide_tail = arith.extui %tail : tensor<$Cxi32> to tensor<$Cxi64>
        %scale = tt.splat $INVERSE : i64 -> tensor<$Cxi64>
        %scaled = arith.muli %wide_tail, %scale : tensor<$Cxi64>
        %modulus = arith.constant dense<$P> : tensor<$Cxi64>
        %residues = arith.remui %scaled, %modulus : tensor<$Cxi64>
        %normalized = arith.trunci %residues : tensor<$Cxi64> to tensor<$Cxi32>
        tt.store %dest_ptrs, %normalized, %live : tensor<$Cx!tt.ptr<i32>>
)";

std::string pivot(int prime, bool row_basis) {
    std::string body = std::string(header);
    // In row-basis mode %status counts pivots, and %flag is the previous
    // pivot row or -1. A missing column leaves subsequent columns eligible.
    body += row_basis
        ? "    %flag = tt.addptr %state, %one : !tt.ptr<i32>, i32\n    %enabled = arith.constant true\n"
        : "    %enabled = arith.cmpi sge, %status, %zero : i32\n";
    body += search;
    // Fermat inversion. Specializing these prime-dependent bit products
    // lets both CUDA and HIP lower the residues by a constant divisor.
    std::string base = "%base", inverse = "%unit";
    for (int bits = prime-2, k = 0; bits; bits >>= 1, ++k) {
        const auto n = std::to_string(k);
        if (bits & 1) {
            body += "        %mul"+n+" = arith.muli "+inverse+", "+base+" : i64\n";
            inverse = "%inv"+n;
            body += "        "+inverse+" = arith.remui %mul"+n+", %prime : i64\n";
        }
        if (bits > 1) {
            body += "        %square"+n+" = arith.muli "+base+", "+base+" : i64\n";
            base = "%power"+n;
            body += "        "+base+" = arith.remui %square"+n+", %prime : i64\n";
        }
    }
    body += normalize;
    if (row_basis) body += R"(        tt.store %flag, %target : !tt.ptr<i32>
        %next = arith.addi %target, %one : i32
        tt.store %state, %next : !tt.ptr<i32>
)";
    body += "      }\n    }\n    tt.return\n  }\n}\n";
    if (!row_basis) substitute(body, "%flag", "%state");
    substitute(body, "%target", row_basis ? "%status" : "%col");
    substitute(body, "$INVERSE", inverse);
    return body;
}

constexpr std::string_view eliminate = R"(    scf.if %enabled {
      %first_row = arith.addi %target, %one : i32
      %first_col = arith.addi %col, %one : i32
      %height = arith.constant $R : i32
      %width = arith.constant $C : i32
      %tile_row = tt.get_program_id x : i32
      %tile_col = tt.get_program_id y : i32
      %row_step = arith.muli %tile_row, %height : i32
      %col_step = arith.muli %tile_col, %width : i32
      %top = arith.addi %first_row, %row_step : i32
      %left = arith.addi %first_col, %col_step : i32
      %lane = tt.make_range {end = $N : i32, start = 0 : i32} : tensor<$Nxi32>
      %tile_width = arith.constant dense<$C> : tensor<$Nxi32>
      %dr = arith.divui %lane, %tile_width : tensor<$Nxi32>
      %dc = arith.remui %lane, %tile_width : tensor<$Nxi32>
      %tops = tt.splat %top : i32 -> tensor<$Nxi32>
      %lefts = tt.splat %left : i32 -> tensor<$Nxi32>
      %i = arith.addi %tops, %dr : tensor<$Nxi32>
      %j = arith.addi %lefts, %dc : tensor<$Nxi32>
      %nrows = tt.splat %rows : i32 -> tensor<$Nxi32>
      %ncols = tt.splat %cols : i32 -> tensor<$Nxi32>
      %row_live = arith.cmpi slt, %i, %nrows : tensor<$Nxi32>
      %col_live = arith.cmpi slt, %j, %ncols : tensor<$Nxi32>
      %active = arith.andi %row_live, %col_live : tensor<$Nxi1>
      %row_offsets = arith.muli %i, %ncols : tensor<$Nxi32>
      %bases = tt.splat %matrix : !tt.ptr<i32> -> tensor<$Nx!tt.ptr<i32>>
      %row_ptrs = tt.addptr %bases, %row_offsets : tensor<$Nx!tt.ptr<i32>>, tensor<$Nxi32>
      %entry_ptrs = tt.addptr %row_ptrs, %j : tensor<$Nx!tt.ptr<i32>>, tensor<$Nxi32>
      %pivot_idx = tt.splat %col : i32 -> tensor<$Nxi32>
      %factor_ptrs = tt.addptr %row_ptrs, %pivot_idx : tensor<$Nx!tt.ptr<i32>>, tensor<$Nxi32>
      %offset = arith.muli %target, %cols : i32
      %pivot_row = tt.addptr %matrix, %offset : !tt.ptr<i32>, i32
      %pivot_bases = tt.splat %pivot_row : !tt.ptr<i32> -> tensor<$Nx!tt.ptr<i32>>
      %pivot_ptrs = tt.addptr %pivot_bases, %j : tensor<$Nx!tt.ptr<i32>>, tensor<$Nxi32>
      %zeros = arith.constant dense<0> : tensor<$Nxi32>
      %entry = tt.load %entry_ptrs, %active, %zeros : tensor<$Nx!tt.ptr<i32>>
      %pivot = tt.load %pivot_ptrs, %col_live, %zeros : tensor<$Nx!tt.ptr<i32>>
      %factor = tt.load %factor_ptrs, %row_live, %zeros : tensor<$Nx!tt.ptr<i32>>
      %wide_pivot = arith.extui %pivot : tensor<$Nxi32> to tensor<$Nxi64>
      %wide_factor = arith.extui %factor : tensor<$Nxi32> to tensor<$Nxi64>
      %product = arith.muli %wide_pivot, %wide_factor : tensor<$Nxi64>
      %modulus = arith.constant dense<$P> : tensor<$Nxi64>
      %residue = arith.remui %product, %modulus : tensor<$Nxi64>
      %subtrahend = arith.trunci %residue : tensor<$Nxi64> to tensor<$Nxi32>
      %diff = arith.subi %entry, %subtrahend : tensor<$Nxi32>
      %negative = arith.cmpi slt, %diff, %zeros : tensor<$Nxi32>
      %prime = arith.constant dense<$P> : tensor<$Nxi32>
      %adjusted = arith.addi %diff, %prime : tensor<$Nxi32>
      %result = arith.select %negative, %adjusted, %diff : tensor<$Nxi1>, tensor<$Nxi32>
      tt.store %entry_ptrs, %result, %active : tensor<$Nx!tt.ptr<i32>>
    }
    tt.return
  }
}
)";

bool block_size(int block, int limit) {
    return block >= 1 && block <= limit && (block & (block-1)) == 0;
}

} // namespace

std::string triton_rank_ir(const std::string& name, int prime, int rows_block, int cols_block, bool row_basis) {
    if (prime != 1000000007 && prime != 1000000009) throw std::invalid_argument("Unsupported modular rank prime.");
    if (!block_size(rows_block, 16384) || !block_size(cols_block, 512))
        throw std::invalid_argument("Unsupported modular rank block.");
    std::string body;
    if (name == "rank_pivot") {
        body = pivot(prime, row_basis);
    } else if (name == "rank_eliminate") {
        if (rows_block * cols_block > 1024) throw std::invalid_argument("Unsupported elimination tile.");
        body = std::string(header);
        if (row_basis) body += R"(    %flag = tt.addptr %state, %one : !tt.ptr<i32>, i32
    %target = tt.load %flag : !tt.ptr<i32>
    %enabled = arith.cmpi sge, %target, %zero : i32
)";
        else body += "    %enabled = arith.cmpi sge, %status, %zero : i32\n";
        body += eliminate;
        if (!row_basis) substitute(body, "%target", "%col");
        substitute(body, "$N", std::to_string(rows_block * cols_block));
    } else {
        throw std::invalid_argument("Unknown modular rank kernel.");
    }
    substitute(body, "$KERNEL", name);
    substitute(body, "$P", std::to_string(prime));
    substitute(body, "$R", std::to_string(rows_block));
    substitute(body, "$C", std::to_string(cols_block));
    return body;
}

} // namespace atlas
