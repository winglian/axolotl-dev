#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::epilogue::operators {

// Identity also provides shared parameters, lifecycle hooks and default BF16 conversion.
// NOTES: the operators do not compose with each other
struct Identity {
    const EpilogueOperatorArgs& args;

    static constexpr bool kIsFP8CastWithClamp = false;
    static constexpr bool kWithStochasticRounding = false;
    static constexpr bool kWithSwiGLUBackward = false;
    static constexpr bool kWithSwiGLUForward = false;
    static constexpr bool kWithReadiness = false;
    static constexpr bool kIncrementalReadiness = false;
    static constexpr bool kWithScatter = false;

    CUTLASS_DEVICE explicit Identity(const EpilogueOperatorArgs& args): args(args) {}

    CUTLASS_DEVICE void init(const uint32_t&, const uint32_t&) {}

    CUTLASS_DEVICE void finalize() const {}

    template <uint32_t kNumValues>
    CUTLASS_DEVICE void apply_values(uint32_t (&)[kNumValues]) const {}

    template <bool kSwapAB>
    CUTLASS_DEVICE void cast_bf16(const uint32_t (&values)[8], uint32_t (&packed)[4],
                                  const uint32_t&, const uint32_t&) const {
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < 4; ++ pair_idx)
            packed[pair_idx] = math::cast_into_bf16_and_pack(values[pair_idx * 2], values[pair_idx * 2 + 1]);
    }
};

// Scale only the product term of a BLAS-style linear combination. Supporting an
// arbitrary beta requires a separate C load/initialization path and does not belong
// in this value-only operator.
template <typename Operator, bool kIncremental = false>
struct Ready: Operator {
    using Operator::Operator;
    static constexpr bool kWithReadiness = true;
    static constexpr bool kIncrementalReadiness = kIncremental;
};

struct ScaleByAlpha: Identity {
    using Identity::Identity;

    template <uint32_t kNumValues>
    CUTLASS_DEVICE void apply_values(uint32_t (&values)[kNumValues]) const {
        DG_STATIC_ASSERT(kNumValues % 2 == 0, "Alpha scaling requires float2 alignment");
        const auto values_f32x2 = reinterpret_cast<float2*>(values);
        const auto alpha_f32x2 = make_float2(args.alpha, args.alpha);
        #pragma unroll
        for (uint32_t value_idx = 0; value_idx < kNumValues / 2; ++ value_idx)
            values_f32x2[value_idx] = __fmul2_rn(values_f32x2[value_idx], alpha_f32x2);
    }
};

// Interleaved gate/up columns keep each activation pair within one accumulator tile.
template <bool kRecomputeBackward = false>
struct SwiGLUForwardBase: Identity {
    using Identity::Identity;
    static constexpr bool kWithSwiGLUForward = true;

    template <uint32_t kLaneBit, uint32_t kRegisterBit>
    CUTLASS_DEVICE static void exchange(uint32_t (&values)[8], uint32_t lane) {
        uint32_t next[8];
        #pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            const auto other = __shfl_xor_sync(0xffffffffu, values[i ^ kRegisterBit], kLaneBit);
            next[i] = static_cast<bool>(lane & kLaneBit) != static_cast<bool>(i & kRegisterBit)
                ? other : values[i];
        }
        #pragma unroll
        for (uint32_t i = 0; i < 8; ++i)
            values[i] = next[i];
    }

    template <bool kSwapAB>
    CUTLASS_DEVICE void apply_forward(const uint32_t (&values)[8], uint32_t row,
                                      uint32_t col, bool valid) const {
        if constexpr (kSwapAB) {
            const auto lane = ptx::get_lane_idx();
            uint32_t transposed[8], contiguous[8];
            #pragma unroll
            for (uint32_t i = 0; i < 8; ++i)
                transposed[i] = values[i];
            // Move column bits into registers, then group four adjacent stores per row.
            exchange<4, 1>(transposed, lane);
            exchange<8, 2>(transposed, lane);
            exchange<16, 4>(transposed, lane);
            const auto source_lane = (lane >> 3) | (lane & 4) | ((lane & 3) << 3);
            #pragma unroll
            for (uint32_t i = 0; i < 8; ++i)
                contiguous[i] = __shfl_sync(0xffffffffu, transposed[i], source_lane);
            const auto base_row = row - 2 * (lane % 4);
            const auto base_col = col - (lane / 16 * 4 + lane / 4 % 4);
            apply_forward<false>(contiguous, base_row + lane / 4, base_col + lane % 4 * 8, valid);
        } else if (valid and row < args.shape_m and col < args.shape_n) {
            const auto inner = args.shape_n / 2;
            const auto pair = col / 2;
            const auto score = args.swiglu_scores[row];
            uint32_t gate_values[4], up_values[4], h_values[4];
            float dscore = 0.0f;
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++i) {
                const auto gate = __bfloat162float(__float2bfloat16_rn(__uint_as_float(values[2 * i])));
                const auto up = __bfloat162float(__float2bfloat16_rn(__uint_as_float(values[2 * i + 1])));
                const float sigmoid = 1.0f / (1.0f + expf(-gate));
                gate_values[i] = __float_as_uint(gate);
                up_values[i] = __float_as_uint(up);
                h_values[i] = __float_as_uint(((gate * sigmoid) * up) * score);
                if constexpr (kRecomputeBackward) {
                    const auto dh = __bfloat162float(static_cast<const nv_bfloat16*>(args.swiglu_dh)[
                        static_cast<uint64_t>(row) * inner + pair + i]);
                    const float silu = gate * sigmoid;
                    gate_values[i] = __float_as_uint(dh * score * up * (sigmoid + silu * (1.0f - sigmoid)));
                    up_values[i] = __float_as_uint(dh * score * silu);
                    dscore += dh * silu * up;
                }
            }
            const auto pack = [](const uint32_t (&v)[4]) {
                return make_uint2(math::cast_into_bf16_and_pack(v[0], v[1]),
                                  math::cast_into_bf16_and_pack(v[2], v[3]));
            };
            auto* z = static_cast<nv_bfloat16*>(kRecomputeBackward ? args.swiglu_dz : args.swiglu_forward_z);
            if constexpr (kRecomputeBackward)
                args.swiglu_ds[static_cast<uint64_t>(row) * (inner / 4) + pair / 4] = dscore;
            auto* h = static_cast<nv_bfloat16*>(args.swiglu_h);
            const auto offset = static_cast<uint64_t>(row) * args.shape_n + pair;
            *reinterpret_cast<uint2*>(z + offset) = pack(gate_values);
            *reinterpret_cast<uint2*>(z + offset + inner) = pack(up_values);
            *reinterpret_cast<uint2*>(h + static_cast<uint64_t>(row) * inner + pair) = pack(h_values);
        }
    }
};

using SwiGLUForward = SwiGLUForwardBase<false>;
using SwiGLURecomputeBackward = SwiGLUForwardBase<true>;

// Axolotl extension: round dH to BF16 before applying the weighted SwiGLU derivative.
struct SwiGLUBackward: Identity {
    using Identity::Identity;
    static constexpr bool kWithSwiGLUBackward = true;

    template <bool kSwapAB>
    CUTLASS_DEVICE void apply_swiglu(uint32_t (&values)[8], const uint32_t row,
                                    const uint32_t col, const uint32_t lane) const {
        const auto z = static_cast<const nv_bfloat16*>(args.swiglu_z);
        const auto dz = static_cast<nv_bfloat16*>(args.swiglu_dz);
        float total = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            const auto r = row + (kSwapAB ? i % 2 : 0);
            const auto c = col + (kSwapAB ? i / 2 * 8 : i);
            float ds = 0.0f;
            float h = 0.0f;
            if (r < args.shape_m and c < args.shape_n) {
                const auto offset = static_cast<uint64_t>(r) * (2 * args.shape_n) + c;
                const float gate = __bfloat162float(z[offset]);
                const float up = __bfloat162float(z[offset + args.shape_n]);
                const float score = args.swiglu_scores[r];
                const float dh = __bfloat162float(__float2bfloat16_rn(__uint_as_float(values[i])));
                const float sigmoid = 1.0f / (1.0f + expf(-gate));
                const float silu = gate * sigmoid;
                h = silu * up * score;
                dz[offset] = __float2bfloat16_rn(dh * score * up * (sigmoid + silu * (1.0f - sigmoid)));
                dz[offset + args.shape_n] = __float2bfloat16_rn(dh * score * silu);
                ds = dh * silu * up;
            }
            values[i] = __float_as_uint(h);
            if constexpr (kSwapAB) {
                #pragma unroll
                for (uint32_t mask = 4; mask <= 16; mask <<= 1)
                    ds += __shfl_xor_sync(0xffffffffu, ds, mask);
                if (lane < 4 and r < args.shape_m and c < args.shape_n)
                    args.swiglu_ds[static_cast<uint64_t>(r) * (args.shape_n / 8) + c / 8] = ds;
            } else {
                total += ds;
            }
        }
        if constexpr (not kSwapAB) {
            if (row < args.shape_m and col < args.shape_n)
                args.swiglu_ds[static_cast<uint64_t>(row) * (args.shape_n / 8) + col / 8] = total;
        }
    }
};

template <bool kAssign, typename Output = float, bool kPeer = false>
struct ScatterOutput: Identity {
    using Identity::Identity;
    static constexpr bool kWithScatter = true;
    uint32_t thread_idx = 0;

    CUTLASS_DEVICE void init(const uint32_t&, const uint32_t& index) {
        thread_idx = index;
    }

    template <uint32_t kLaneBit, uint32_t kRegisterBit>
    CUTLASS_DEVICE static void exchange(uint32_t (&values)[8], uint32_t lane) {
        uint32_t next[8];
        #pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            const auto other = __shfl_xor_sync(0xffffffffu, values[i ^ kRegisterBit], kLaneBit);
            next[i] = static_cast<bool>(lane & kLaneBit) != static_cast<bool>(i & kRegisterBit)
                ? other : values[i];
        }
        #pragma unroll
        for (uint32_t i = 0; i < 8; ++i)
            values[i] = next[i];
    }

    template <bool kSwapAB, uint32_t kValues>
    CUTLASS_DEVICE void scatter(const uint32_t (&values)[kValues],
                                const uint32_t row, const uint32_t col) const {
        DG_STATIC_ASSERT(kValues == 8, "Scatter requires eight BF16-output accumulator values");
        const auto lane = ptx::get_lane_idx();
        uint32_t transposed[8];
        #pragma unroll
        for (uint32_t i = 0; i < 8; ++i)
            transposed[i] = values[i];
        // Exchange row bits into registers so adjacent lanes issue adjacent-column atomics.
        if constexpr (kSwapAB) {
            exchange<1, 2>(transposed, lane);
            exchange<2, 4>(transposed, lane);
        } else {
            exchange<1, 1>(transposed, lane);
            exchange<2, 2>(transposed, lane);
            exchange<4, 4>(transposed, lane);
        }
        const auto base_row = row - (kSwapAB ? 2 * (lane % 4) : lane % 8);
        const auto column = kSwapAB
            ? col - (lane / 16 * 4 + lane / 4 % 4) + (lane >> 2) + (lane % 4) * 8
            : col + lane % 8;
        #pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            const auto r = base_row + i;
            if (r < args.shape_m and column < args.shape_n) {
                const auto token = args.scatter_tokens[r];
                DG_TRAP_ONLY_DEVICE_ASSERT(token >= -1 and token < static_cast<int>(args.scatter_rows));
                if (token >= 0) {
                    const auto value = __bfloat162float(__float2bfloat16_rn(__uint_as_float(transposed[i])));
                    Output* destination;
                    if constexpr (kPeer) {
                        const auto rank = token / args.scatter_peer_rows;
                        const auto slot = token % args.scatter_peer_rows;
                        const auto pointers = static_cast<const uint64_t*>(args.scatter_output);
                        destination = reinterpret_cast<Output*>(pointers[rank]) + static_cast<uint64_t>(slot) * args.shape_n + column;
                    } else {
                        destination = static_cast<Output*>(args.scatter_output) + static_cast<uint64_t>(token) * args.shape_n + column;
                    }
                    if constexpr (kAssign)
                        *destination = static_cast<Output>(value);
                    else
                        atomicAdd_system(destination, value);
                }
            }
        }
    }

    template <uint32_t kThreads>
    CUTLASS_DEVICE void publish() const {
        __threadfence_system();
        cutlass::arch::NamedBarrier::sync(kThreads, 1);
        if (thread_idx == 0) {
            auto* counter = args.scatter_queue + args.scatter_task * 4 + 2;
            uint32_t previous;
            asm volatile("atom.acq_rel.sys.global.add.u32 %0, [%1], 1;"
                         : "=r"(previous) : "l"(counter) : "memory");
            if (previous + 1 == gridDim.x * gridDim.y * gridDim.z) {
                auto* flag = counter + 1;
                asm volatile("st.release.sys.global.u32 [%0], 1;" :: "l"(flag) : "memory");
            }
        }
    }
};

using ScatterAdd = ScatterOutput<false>;
using ScatterStore = ScatterOutput<true>;
using ScatterStoreBF16 = ScatterOutput<true, nv_bfloat16>;
using PeerScatterStoreBF16 = ScatterOutput<true, nv_bfloat16, true>;

struct StochasticRoundToBF16: Identity {
private:
    // The quartet hash is the sum of the four members' partial terms plus `quartet_n_idx`;
    // each member then finalizes the sum and selects its own 16-bit random half
    CUTLASS_DEVICE static uint32_t partial_hash(const uint32_t& value, const uint32_t& quartet_offset) {
        return value * (quartet_offset == 0 ? 0x5671d42bu :
                        quartet_offset == 1 ? 0x9995e499u :
                        quartet_offset == 2 ? 0xace1b8a5u : 0xe153538du);
    }

    CUTLASS_DEVICE static uint32_t select_random_bits(uint32_t h, const uint32_t& quartet_offset) {
        h ^= h >> 23;
        h *= 0x7feb352du;
        h ^= h >> 16;
        h *= quartet_offset < 2 ? 0x846ca68bu : 0xd35a2d97u;
        h ^= h >> 11;
        return (h >> (quartet_offset % 2 * 16)) & 0xffffu;
    }

public:
    using Identity::Identity;
    static constexpr bool kWithStochasticRounding = true;

    // N-quartets are lane-local normally, or span lanes spaced by four with swap-AB.
    template <bool kSwapAB>
    CUTLASS_DEVICE void cast_bf16(const uint32_t (&values)[8], uint32_t (&packed)[4],
                                  const uint32_t& n_idx, const uint32_t& lane_idx) const {
        const auto quartet_offset = lane_idx / 4 % 4;
        uint32_t hashes[8];
        #pragma unroll
        for (uint32_t value_idx = 0; value_idx < 8; ++ value_idx)
            hashes[value_idx] = partial_hash(values[value_idx], kSwapAB ? quartet_offset : value_idx % 4);
        if constexpr (kSwapAB) {
            #pragma unroll
            for (uint32_t xor_mask = 4; xor_mask <= 8; xor_mask <<= 1) {
                #pragma unroll
                for (uint32_t value_idx = 0; value_idx < 8; ++ value_idx)
                    hashes[value_idx] += __shfl_xor_sync(0xffffffffu, hashes[value_idx], xor_mask);
            }
        }
        #pragma unroll
        for (uint32_t value_idx = 0; value_idx < 8; value_idx += 2) {
            auto hash_lower = hashes[value_idx], hash_upper = hashes[value_idx + 1];
            if constexpr (not kSwapAB) {
                const auto quartet_idx = value_idx / 4 * 4;
                hash_lower = hash_upper = hashes[quartet_idx] + hashes[quartet_idx + 1] +
                                          hashes[quartet_idx + 2] + hashes[quartet_idx + 3];
            }
            const auto quartet_n_idx = n_idx / 4 + (kSwapAB ? value_idx + lane_idx / 16 : value_idx / 4);
            const auto offset = kSwapAB ? quartet_offset : value_idx % 4;
            packed[value_idx / 2] = ptx::cvt_rs_bf16x2_f32(
                *reinterpret_cast<const float*>(&values[value_idx]),
                *reinterpret_cast<const float*>(&values[value_idx + 1]),
                select_random_bits(hash_lower + quartet_n_idx, offset) |
                select_random_bits(hash_upper + quartet_n_idx, offset + (kSwapAB ? 0 : 1)) << 16);
        }
    }
};

// Quantize D into FP8 with dynamic per-row, per-32-column UE8M0 SFs, packed into `uint32_t`
// words in a TMA-aligned MN-major layout (the same layout accepted for SFA)
// NOTES: the accumulator is rounded into BF16 before the amax/scale/cast steps, so the
//        output bitwise matches a BF16 D followed by the standalone per-token cast kernel
struct CastToFP8WithClamp: Identity {
private:
    uint32_t num_clamped = 0;
    bool count_enabled = false;

public:
    static constexpr bool kIsFP8CastWithClamp = true;
    static constexpr uint32_t kSFGranN = 32;

    using Identity::Identity;

    // Reset while the MMA pipeline starts; idle CTAs need no counter work.
    CUTLASS_DEVICE void init(const uint32_t& num_blocks, const uint32_t& epilogue_thread_idx) {
        count_enabled = args.clamped_count != nullptr and blockIdx.x < num_blocks;
        if (not count_enabled or epilogue_thread_idx != 0)
            return;
        const auto grid_tag = ptx::get_grid_idx() + 1;
        const auto ready_ptr = reinterpret_cast<decltype(grid_tag)*>(args.clamp_count_ready);
        if (blockIdx.x == 0) {
            *args.clamped_count = 0;
            ptx::st_rel(ready_ptr, grid_tag);
        } else {
            while (ptx::ld_acq_gpu(ready_ptr) != grid_tag) {}
        }
    }

    // Round and count BF16 values, then compute SFs for the chosen TMEM layout.
    template <bool kSwapAB, uint32_t kNumValues>
    CUTLASS_DEVICE void prepare_fp8(const uint32_t (&values)[kNumValues],
                                    nv_bfloat162 (&values_bf16x2)[kNumValues / 2], uint2& sf_exp,
                                    const uint32_t& row_idx, const uint32_t& col_idx) {
        // An infinite bound makes the unclamped path use the same conversion loop.
        const auto bound = __float2bfloat162_rn(
            args.clamp_value != 0.0f ? args.clamp_value : cute::numeric_limits<float>::infinity());
        // Count while rounding to avoid a runtime branch in the unrolled loop.
        uint32_t packed_count = 0;
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumValues / 2; ++ pair_idx) {
            const auto pair = math::cast_into_bf16x2(values[pair_idx * 2], values[pair_idx * 2 + 1]);
            values_bf16x2[pair_idx] = pair;
            packed_count += __hgtu2_mask(__habs2(pair), bound) & 0x00010001u;
        }

        // Two independent 16-bit counts, one per BF16 component. N is SF-group aligned.
        DG_STATIC_ASSERT(kNumValues / 2 < 65536, "Packed component counts must not overflow");
        if (args.clamp_value != 0.0f and col_idx < args.shape_n) {
            num_clamped += row_idx < args.shape_m ? packed_count & 0xffffu : 0u;
            num_clamped += row_idx + (kSwapAB ? 1 : 0) < args.shape_m ? packed_count >> 16 : 0u;
        }

        nv_bfloat162 amax;
        if constexpr (kSwapAB) {
            amax = math::warp_reduce_max<4, true>(
                __hmax2(__hmax2(__habs2(values_bf16x2[0]), __habs2(values_bf16x2[1])),
                        __hmax2(__habs2(values_bf16x2[2]), __habs2(values_bf16x2[3]))));
        } else {
            amax = __bfloat162bfloat162(math::get_packed_bf16_amax(values_bf16x2));
        }
        // Clamp amax once; defer element clamping until packing so the reduction can start earlier.
        amax = __hmin2_nan(amax, bound);
        sf_exp.x = math::get_ue8m0_sf_exp(amax.x);
        sf_exp.y = kSwapAB ? math::get_ue8m0_sf_exp(amax.y) : sf_exp.x;
    }

    // Clamp, scale and pack a prepared BF16 fragment.
    template <uint32_t kNumPairs>
    CUTLASS_DEVICE void cast_fp8(const nv_bfloat162 (&values_bf16x2)[kNumPairs],
                                 uint32_t (&packed)[kNumPairs / 2], const uint2& sf_exp) const {
        const auto bound = __float2bfloat162_rn(
            args.clamp_value != 0.0f ? args.clamp_value : cute::numeric_limits<float>::infinity());
        const auto sf_inv = nv_bfloat162(math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp.x),
                                         math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp.y));
        #pragma unroll
        for (uint32_t packed_idx = 0; packed_idx < kNumPairs / 2; ++ packed_idx) {
            const auto lower = ptx::clamp_bf16x2(values_bf16x2[packed_idx * 2], bound);
            const auto upper = ptx::clamp_bf16x2(values_bf16x2[packed_idx * 2 + 1], bound);
            packed[packed_idx] = math::scale_bf16x2_into_fp8x4(lower, upper, sf_inv, sf_inv);
        }
    }

    // One non-returning atomic per warp.
    CUTLASS_DEVICE void finalize() const {
        if (count_enabled) {
            const auto warp_count = __reduce_add_sync(0xffffffffu, num_clamped);
            if (ptx::get_lane_idx() == 0)
                ptx::red_add(args.clamped_count, warp_count);
        }
    }

    // Store one SF byte (`uint8_t`), or all four SF bytes of one packed word at once
    // (`uint32_t`, with word-aligned `group_n_idx` and `shape_n % 128 == 0`, so that a word
    // never crosses a batch or shape boundary)
    // NOTES: batched GEMMs flatten the SF columns of all batches along `(batch_idx, n)`,
    //        matching a D viewed by the consumer as a `(shape_m, num_batches * shape_n)` 2D tensor
    template <typename sf_t>
    CUTLASS_DEVICE void store_sf(const uint32_t& row_idx, const uint32_t& group_n_idx,
                                 const uint32_t& batch_idx, const sf_t& sf) const {
        if (row_idx >= args.shape_m or group_n_idx >= args.shape_n)
            return;
        const auto sf_idx = (batch_idx * args.shape_n + group_n_idx) / kSFGranN;
        const auto sf_word_ptr = args.sfd + (sf_idx / 4) * args.sfd_stride + row_idx;
        const auto sf_ptr = reinterpret_cast<uint8_t*>(sf_word_ptr) + sf_idx % 4;
        *reinterpret_cast<sf_t*>(sf_ptr) = sf;
    }
};

} // namespace deep_gemm::epilogue::operators
