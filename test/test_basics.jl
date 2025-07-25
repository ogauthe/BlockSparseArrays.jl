using Adapt: adapt
using ArrayLayouts: zero!
using BlockArrays:
  BlockArrays,
  Block,
  BlockArray,
  BlockRange,
  BlockVector,
  BlockedOneTo,
  BlockedArray,
  BlockedVector,
  blockedrange,
  blocklength,
  blocklengths,
  blocksize,
  blocksizes,
  mortar,
  undef_blocks
using BlockSparseArrays:
  @view!,
  BlockSparseArray,
  BlockSparseMatrix,
  BlockSparseVector,
  BlockType,
  BlockView,
  blockdiagindices,
  blockreshape,
  blocksparse,
  blocksparsezeros,
  blockstoredlength,
  blockstype,
  blocktype,
  eachblockstoredindex,
  eachstoredblock,
  eachstoredblockdiagindex,
  similartype_unchecked,
  sparsemortar,
  view!
using GPUArraysCore: @allowscalar
using JLArrays: JLArray, JLMatrix
using LinearAlgebra: Adjoint, Transpose, dot, norm, tr
using SparseArraysBase:
  SparseArrayDOK, SparseMatrixDOK, SparseVectorDOK, isstored, storedlength
using Test: @test, @test_broken, @test_throws, @testset, @inferred
using TestExtras: @constinferred
using TypeParameterAccessors: TypeParameterAccessors, Position
include("TestBlockSparseArraysUtils.jl")

elts = (Float32, Float64, ComplexF32)
arrayts = (Array, JLArray)
@testset "BlockSparseArrays basics (arraytype=$arrayt, eltype=$elt)" for arrayt in arrayts,
  elt in elts

  dev = adapt(arrayt)

  @testset "Broken" begin
    # TODO: Fix this and turn it into a proper test.
    a = dev(BlockSparseArray{elt}(undef, [2, 3], [2, 3]))
    a[Block(1, 1)] = dev(randn(elt, 2, 2))
    a[Block(2, 2)] = dev(randn(elt, 3, 3))
    @test_broken a[:, 4]

    # TODO: Fix this and turn it into a proper test.
    a = dev(BlockSparseArray{elt}(undef, [2, 3], [2, 3]))
    a[Block(1, 1)] = dev(randn(elt, 2, 2))
    a[Block(2, 2)] = dev(randn(elt, 3, 3))
    @allowscalar @test a[2:4, 4] == Array(a)[2:4, 4]
    @test_broken a[4, 2:4]

    @test a[Block(1), :] isa BlockSparseArray{elt}
    @test adjoint(a) isa Adjoint{elt,<:BlockSparseArray}
    @test_broken adjoint(a)[Block(1), :] isa Adjoint{elt,<:BlockSparseArray}
    # could also be directly a BlockSparseArray
  end
  @testset "Constructors" begin
    # BlockSparseMatrix
    bs = ([2, 3], [3, 4])
    for T in (
      BlockSparseArray{elt},
      BlockSparseArray{elt,2},
      BlockSparseMatrix{elt},
      BlockSparseArray{elt,2,Matrix{elt}},
      BlockSparseMatrix{elt,Matrix{elt}},
      ## BlockSparseArray{elt,2,Matrix{elt},SparseMatrixDOK{Matrix{elt}}}, # TODO
      ## BlockSparseMatrix{elt,Matrix{elt},SparseMatrixDOK{Matrix{elt}}}, # TODO
    )
      for args in (
        (undef, bs),
        (undef, bs...),
        (undef, blockedrange.(bs)),
        (undef, blockedrange.(bs)...),
      )
        a = T(args...)
        @test eltype(a) == elt
        @test blocktype(a) == Matrix{elt}
        @test blockstype(a) <: SparseMatrixDOK{Matrix{elt}}
        @test blocklengths.(axes(a)) == ([2, 3], [3, 4])
        @test iszero(a)
        @test iszero(blockstoredlength(a))
        @test iszero(storedlength(a))
      end
      T != BlockSparseArray{elt} && @test_throws ArgumentError T(undef, bs[1:1])
    end

    # BlockSparseVector
    bs = ([2, 3],)
    for T in (
      BlockSparseArray{elt},
      BlockSparseArray{elt,1},
      BlockSparseVector{elt},
      BlockSparseArray{elt,1,Vector{elt}},
      BlockSparseVector{elt,Vector{elt}},
      ## BlockSparseArray{elt,1,Vector{elt},SparseVectorDOK{Vector{elt}}}, # TODO
      ## BlockSparseVector{elt,Vector{elt},SparseVectorDOK{Vector{elt}}}, # TODO
    )
      for args in (
        (undef, bs),
        (undef, bs...),
        (undef, blockedrange.(bs)),
        (undef, blockedrange.(bs)...),
      )
        a = T(args...)
        @test eltype(a) == elt
        @test blocktype(a) == Vector{elt}
        @test blockstype(a) <: SparseVectorDOK{Vector{elt}}
        @test blocklengths.(axes(a)) == ([2, 3],)
        @test iszero(a)
        @test iszero(blockstoredlength(a))
        @test iszero(storedlength(a))
      end
    end

    for dims in (
      ([2, 2], [2, 2]),
      (([2, 2], [2, 2]),),
      blockedrange.(([2, 2], [2, 2])),
      (blockedrange.(([2, 2], [2, 2])),),
    )
      @test_throws ArgumentError BlockSparseVector{elt}(undef, dims...)
    end

    # Convenient constructors.
    a = blocksparsezeros([2, 3], [2, 3])
    @test iszero(a)
    @test iszero(blockstoredlength(a))
    @test a isa BlockSparseMatrix{Float64,Matrix{Float64}}
    @test blocktype(a) == Matrix{Float64}
    @test blockstype(a) <: SparseMatrixDOK{Matrix{Float64}}
    @test blocksize(a) == (2, 2)
    @test blocksizes(a) == [(2, 2) (2, 3); (3, 2) (3, 3)]

    a = blocksparsezeros(elt, [2, 3], [2, 3])
    @test iszero(a)
    @test iszero(blockstoredlength(a))
    @test a isa BlockSparseMatrix{elt,Matrix{elt}}
    @test blocktype(a) == Matrix{elt}
    @test blockstype(a) <: SparseMatrixDOK{Matrix{elt}}
    @test blocksize(a) == (2, 2)
    @test blocksizes(a) == [(2, 2) (2, 3); (3, 2) (3, 3)]

    a = blocksparsezeros(BlockType(arrayt{elt,2}), [2, 3], [2, 3])
    @test iszero(a)
    @test iszero(blockstoredlength(a))
    @test a isa BlockSparseMatrix{elt,arrayt{elt,2}}
    @test blocktype(a) == arrayt{elt,2}
    @test blockstype(a) <: SparseMatrixDOK{arrayt{elt,2}}
    @test blocksize(a) == (2, 2)
    @test blocksizes(a) == [(2, 2) (2, 3); (3, 2) (3, 3)]

    d = Dict(Block(1, 1) => dev(randn(elt, 2, 2)), Block(2, 2) => dev(randn(elt, 3, 3)))
    a = blocksparse(d, [2, 3], [2, 3])
    @test !iszero(a)
    @test a[Block(1, 1)] == d[Block(1, 1)]
    @test a[Block(2, 2)] == d[Block(2, 2)]
    @test blockstoredlength(a) == 2
    @test a isa BlockSparseMatrix{elt,arrayt{elt,2}}
    @test blocktype(a) == arrayt{elt,2}
    @test blockstype(a) <: SparseMatrixDOK{arrayt{elt,2}}
    @test blocksize(a) == (2, 2)
    @test blocksizes(a) == [(2, 2) (2, 3); (3, 2) (3, 3)]
  end
  @testset "blockstype, blocktype" begin
    a = arrayt(randn(elt, 2, 2))
    @test (@constinferred blockstype(a)) <: BlockArrays.BlocksView{elt,2}
    # TODO: This is difficult to determine just from type information.
    @test_broken blockstype(typeof(a)) <: BlockArrays.BlocksView{elt,2}
    @test (@constinferred blocktype(a)) <: SubArray{elt,2,arrayt{elt,2}}
    # TODO: This is difficult to determine just from type information.
    @test_broken blocktype(typeof(a)) <: SubArray{elt,2,arrayt{elt,2}}

    a = BlockSparseMatrix{elt,arrayt{elt,2}}(undef, [1, 1], [1, 1])
    @test (@constinferred blockstype(a)) <: SparseMatrixDOK{arrayt{elt,2}}
    @test (@constinferred blockstype(typeof(a))) <: SparseMatrixDOK{arrayt{elt,2}}
    @test (@constinferred blocktype(a)) <: arrayt{elt,2}
    @test (@constinferred blocktype(typeof(a))) <: arrayt{elt,2}

    a = BlockArray(arrayt(randn(elt, (2, 2))), [1, 1], [1, 1])
    @test (@constinferred blockstype(a)) === Matrix{arrayt{elt,2}}
    @test (@constinferred blockstype(typeof(a))) === Matrix{arrayt{elt,2}}
    @test (@constinferred blocktype(a)) <: arrayt{elt,2}
    @test (@constinferred blocktype(typeof(a))) <: arrayt{elt,2}

    a = BlockedArray(arrayt(randn(elt, 2, 2)), [1, 1], [1, 1])
    @test (@constinferred blockstype(a)) <: BlockArrays.BlocksView{elt,2}
    # TODO: This is difficult to determine just from type information.
    @test_broken blockstype(typeof(a)) <: BlockArrays.BlocksView{elt,2}
    @test (@constinferred blocktype(a)) <: SubArray{elt,2,arrayt{elt,2}}
    # TODO: This is difficult to determine just from type information.
    @test_broken blocktype(typeof(a)) <: SubArray{elt,2,arrayt{elt,2}}

    # sparsemortar
    for ax in (
      ([2, 3], [2, 3]),
      (([2, 3], [2, 3]),),
      blockedrange.(([2, 3], [2, 3])),
      (blockedrange.(([2, 3], [2, 3])),),
    )
      blocks = SparseArrayDOK{arrayt{elt,2}}(undef_blocks, ax...)
      blocks[2, 1] = arrayt(randn(elt, 3, 2))
      blocks[1, 2] = arrayt(randn(elt, 2, 3))
      a = sparsemortar(blocks, ax...)
      @test a isa BlockSparseArray{elt,2,arrayt{elt,2}}
      @test iszero(a[Block(1, 1)])
      @test a[Block(2, 1)] == blocks[2, 1]
      @test a[Block(1, 2)] == blocks[1, 2]
      @test iszero(a[Block(2, 2)])
    end
  end
  @testset "Basics" begin
    a = dev(BlockSparseArray{elt}(undef, [2, 3], [2, 3]))
    @allowscalar @test a == dev(
      BlockSparseArray{elt}(undef, blockedrange([2, 3]), blockedrange([2, 3]))
    )
    @test eltype(a) === elt
    @test axes(a) == (1:5, 1:5)
    @test all(aᵢ -> aᵢ isa BlockedOneTo, axes(a))
    @test blocklength.(axes(a)) == (2, 2)
    @test blocksize(a) == (2, 2)
    @test size(a) == (5, 5)
    @test blockstoredlength(a) == 0
    @test iszero(a)
    @allowscalar @test all(I -> iszero(a[I]), eachindex(a))
    @test_throws DimensionMismatch a[Block(1, 1)] = randn(elt, 2, 3)

    a = BlockSparseArray{elt}(undef, [2, 3], [2, 3])
    a[3, 3] = 33
    @test eltype(a) === elt
    @test axes(a) == (1:5, 1:5)
    @test all(aᵢ -> aᵢ isa BlockedOneTo, axes(a))
    @test blocklength.(axes(a)) == (2, 2)
    @test blocksize(a) == (2, 2)
    @test size(a) == (5, 5)
    @test blockstoredlength(a) == 1
    @test !iszero(a)
    @test a[3, 3] == 33
    @test all(eachindex(a)) do I
      if I == CartesianIndex(3, 3)
        a[I] == 33
      else
        iszero(a[I])
      end
    end

    a = BlockSparseArray{elt}(undef, [2, 3], [2, 3])
    a[Block(2, 1)] = randn(elt, 3, 2)
    a[Block(1, 2)] = randn(elt, 2, 3)
    @test issetequal(eachstoredblock(a), [a[Block(2, 1)], a[Block(1, 2)]])
    @test issetequal(eachblockstoredindex(a), [Block(2, 1), Block(1, 2)])
    @test issetequal(blockdiagindices(a), [Block(1, 1), Block(2, 2)])
    @test isempty(eachstoredblockdiagindex(a))
    @test norm(a) ≈ norm(Array(a))
    for p in 1:3
      @test norm(a, p) ≈ norm(Array(a), p)
    end
    @test tr(a) ≈ tr(Array(a))

    a[3, 3] = NaN
    @test isnan(norm(a))

    a = dev(BlockSparseArray{elt}(undef, [2, 3], [2, 3]))
    a[Block(1, 1)] = dev(randn(elt, 2, 2))
    @test issetequal(eachstoredblockdiagindex(a), [Block(1, 1)])
    @test norm(a) ≈ norm(Array(a))
    for p in 1:3
      @test norm(a, p) ≈ norm(Array(a), p)
    end
    @test tr(a) ≈ tr(Array(a))

    # Empty constructor
    for a in (dev(BlockSparseArray{elt}(undef)),)
      @test size(a) == ()
      @test isone(length(a))
      @test blocksize(a) == ()
      @test blocksizes(a) == fill(())
      @test iszero(blockstoredlength(a))
      @test iszero(@allowscalar(a[]))
      @test iszero(@allowscalar(a[CartesianIndex()]))
      @test a[Block()] == dev(fill(0))
      @test iszero(@allowscalar(a[Block()][]))
      @test iszero(@allowscalar(a[Block()[]]))
      @test Array(a) isa Array{elt,0}
      @test Array(a) == fill(0)
      for b in (
        (b=copy(a); @allowscalar(b[] = 2); b),
        (b=copy(a); @allowscalar(b[CartesianIndex()] = 2); b),
        (b=copy(a); @allowscalar(b[Block()[]] = 2); b),
        # Regression test for https://github.com/ITensor/BlockSparseArrays.jl/issues/27.
        (b=copy(a); b[Block()]=dev(fill(2)); b),
      )
        @test size(b) == ()
        @test isone(length(b))
        @test blocksize(b) == ()
        @test blocksizes(b) == fill(())
        @test isone(blockstoredlength(b))
        @test @allowscalar(b[]) == 2
        @test @allowscalar(b[CartesianIndex()]) == 2
        @test b[Block()] == dev(fill(2))
        @test @allowscalar(b[Block()][]) == 2
        @test @allowscalar(b[Block()[]]) == 2
        @test Array(b) isa Array{elt,0}
        @test Array(b) == fill(2)
      end
    end

    @testset "Transpose" begin
      a = dev(BlockSparseArray{elt}(undef, [2, 2], [3, 3, 1]))
      a[Block(1, 1)] = dev(randn(elt, 2, 3))
      a[Block(2, 3)] = dev(randn(elt, 2, 1))

      at = @inferred transpose(a)
      @test at isa Transpose
      @test size(at) == reverse(size(a))
      @test blocksize(at) == reverse(blocksize(a))
      @test storedlength(at) == storedlength(a)
      @test blockstoredlength(at) == blockstoredlength(a)
      for bind in eachblockstoredindex(a)
        bindt = Block(reverse(Int.(Tuple(bind))))
        @test bindt in eachblockstoredindex(at)
      end

      @test @views(at[Block(1, 1)]) == transpose(a[Block(1, 1)])
      @test @views(at[Block(1, 1)]) isa Transpose
      @test @views(at[Block(3, 2)]) == transpose(a[Block(2, 3)])
      # TODO: BlockView == AbstractArray calls scalar code
      @test @allowscalar @views(at[Block(1, 2)]) == transpose(a[Block(2, 1)])
      @test @views(at[Block(1, 2)]) isa Transpose
    end

    @testset "Adjoint" begin
      a = dev(BlockSparseArray{elt}(undef, [2, 2], [3, 3, 1]))
      a[Block(1, 1)] = dev(randn(elt, 2, 3))
      a[Block(2, 3)] = dev(randn(elt, 2, 1))

      at = @inferred adjoint(a)
      @test at isa Adjoint
      @test size(at) == reverse(size(a))
      @test blocksize(at) == reverse(blocksize(a))
      @test storedlength(at) == storedlength(a)
      @test blockstoredlength(at) == blockstoredlength(a)
      for bind in eachblockstoredindex(a)
        bindt = Block(reverse(Int.(Tuple(bind))))
        @test bindt in eachblockstoredindex(at)
      end

      @test @views(at[Block(1, 1)]) == adjoint(a[Block(1, 1)])
      @test @views(at[Block(1, 1)]) isa Adjoint
      @test @views(at[Block(3, 2)]) == adjoint(a[Block(2, 3)])
      # TODO: BlockView == AbstractArray calls scalar code
      @test @allowscalar @views(at[Block(1, 2)]) == adjoint(a[Block(2, 1)])
      @test @views(at[Block(1, 2)]) isa Adjoint
    end
  end
  @testset "adapt" begin
    a = BlockSparseArray{elt}(undef, [2, 2], [2, 2])
    a_12 = randn(elt, 2, 2)
    a[Block(1, 2)] = a_12
    a_jl = adapt(JLArray, a)
    @test a_jl isa BlockSparseMatrix{elt,JLMatrix{elt}}
    @test blocktype(a_jl) == JLMatrix{elt}
    @test blockstoredlength(a_jl) == 1
    @test a_jl[Block(1, 2)] isa JLMatrix{elt}
    @test adapt(Array, a_jl[Block(1, 2)]) == a_12

    a = BlockSparseArray{elt}(undef, [2, 2], [2, 2])
    a_12 = randn(elt, 2, 2)
    a[Block(1, 2)] = a_12
    a_jl = adapt(JLArray, @view(a[:, :]))
    @test a_jl isa SubArray{elt,2,<:BlockSparseMatrix{elt,JLMatrix{elt}}}
    @test blocktype(a_jl) == JLMatrix{elt}
    @test blockstoredlength(a_jl) == 1
    @test a_jl[Block(1, 2)] isa JLMatrix{elt}
    @test adapt(Array, a_jl[Block(1, 2)]) == a_12
  end
  @testset "LinearAlgebra" begin
    a1 = dev(BlockSparseArray{elt}(undef, [2, 3], [2, 3]))
    a1[Block(1, 1)] = dev(randn(elt, size(@view(a1[Block(1, 1)]))))
    a2 = dev(BlockSparseArray{elt}(undef, [2, 3], [2, 3]))
    a2[Block(1, 1)] = dev(randn(elt, size(@view(a1[Block(1, 1)]))))
    a_dest = a1 * a2
    @allowscalar @test Array(a_dest) ≈ Array(a1) * Array(a2)
    @test a_dest isa BlockSparseArray{elt}
    @test blockstoredlength(a_dest) == 1
  end
  @testset "Matrix multiplication" begin
    a1 = dev(BlockSparseArray{elt}(undef, [2, 3], [2, 3]))
    a1[Block(1, 2)] = dev(randn(elt, size(@view(a1[Block(1, 2)]))))
    a1[Block(2, 1)] = dev(randn(elt, size(@view(a1[Block(2, 1)]))))
    a2 = dev(BlockSparseArray{elt}(undef, [2, 3], [2, 3]))
    a2[Block(1, 2)] = dev(randn(elt, size(@view(a2[Block(1, 2)]))))
    a2[Block(2, 1)] = dev(randn(elt, size(@view(a2[Block(2, 1)]))))
    for (a1′, a2′) in ((a1, a2), (a1', a2), (a1, a2'), (a1', a2'))
      a_dest = a1′ * a2′
      @allowscalar @test Array(a_dest) ≈ Array(a1′) * Array(a2′)
    end
  end
  @testset "Dot product" begin
    a1 = dev(BlockSparseArray{elt}(undef, [2, 3, 4]))
    a1[Block(1)] = dev(randn(elt, size(@view(a1[Block(1)]))))
    a1[Block(3)] = dev(randn(elt, size(@view(a1[Block(3)]))))
    a2 = dev(BlockSparseArray{elt}(undef, [2, 3, 4]))
    a2[Block(2)] = dev(randn(elt, size(@view(a1[Block(2)]))))
    a2[Block(3)] = dev(randn(elt, size(@view(a1[Block(3)]))))
    @test a1' * a2 ≈ Array(a1)' * Array(a2)
    @test dot(a1, a2) ≈ a1' * a2
  end
  @testset "cat" begin
    a1 = dev(BlockSparseArray{elt}(undef, [2, 3], [2, 3]))
    a1[Block(2, 1)] = dev(randn(elt, size(@view(a1[Block(2, 1)]))))
    a2 = dev(BlockSparseArray{elt}(undef, [2, 3], [2, 3]))
    a2[Block(1, 2)] = dev(randn(elt, size(@view(a2[Block(1, 2)]))))

    a_dest = cat(a1, a2; dims=1)
    @test blockstoredlength(a_dest) == 2
    @test blocklengths.(axes(a_dest)) == ([2, 3, 2, 3], [2, 3])
    @test issetequal(eachblockstoredindex(a_dest), [Block(2, 1), Block(3, 2)])
    @test a_dest[Block(2, 1)] == a1[Block(2, 1)]
    @test a_dest[Block(3, 2)] == a2[Block(1, 2)]

    a_dest = cat(a1, a2; dims=2)
    @test blockstoredlength(a_dest) == 2
    @test blocklengths.(axes(a_dest)) == ([2, 3], [2, 3, 2, 3])
    @test issetequal(eachblockstoredindex(a_dest), [Block(2, 1), Block(1, 4)])
    @test a_dest[Block(2, 1)] == a1[Block(2, 1)]
    @test a_dest[Block(1, 4)] == a2[Block(1, 2)]

    a_dest = cat(a1, a2; dims=(1, 2))
    @test blockstoredlength(a_dest) == 2
    @test blocklengths.(axes(a_dest)) == ([2, 3, 2, 3], [2, 3, 2, 3])
    @test issetequal(eachblockstoredindex(a_dest), [Block(2, 1), Block(3, 4)])
    @test a_dest[Block(2, 1)] == a1[Block(2, 1)]
    @test a_dest[Block(3, 4)] == a2[Block(1, 2)]
  end

  @testset "blockreshape" begin
    a = dev(BlockSparseArray{elt}(undef, ([3, 4], [2, 3])))
    a[Block(1, 2)] = dev(randn(elt, size(@view(a[Block(1, 2)]))))
    a[Block(2, 1)] = dev(randn(elt, size(@view(a[Block(2, 1)]))))
    b = blockreshape(a, [6, 8, 9, 12])
    @test reshape(a[Block(1, 2)], 9) == b[Block(3)]
    @test reshape(a[Block(2, 1)], 8) == b[Block(2)]
    @test blockstoredlength(b) == 2
    @test storedlength(b) == 17

    # Zero-dimensional limit (check for ambiguity errors).
    # Regression test for https://github.com/ITensor/BlockSparseArrays.jl/issues/98.
    a = dev(BlockSparseArray{elt}(undef, ()))
    a[Block()] = dev(randn(elt, ()))
    b = blockreshape(a)
    @test a[Block()] == b[Block()]
    @test blockstoredlength(b) == 1
    @test storedlength(b) == 1
  end
  @testset "show" begin
    vectort_elt = arrayt{elt,1}
    matrixt_elt = arrayt{elt,2}
    arrayt_elt = arrayt{elt,3}

    a = BlockSparseVector{elt,arrayt{elt,1}}(undef, [2, 2])
    res = sprint(summary, a)
    function ref_vec(elt, arrayt, prefix="")
      return "2-blocked 4-element $(prefix)BlockSparseVector{$(elt), $(arrayt), …, …}"
    end
    # Either option is possible depending on namespacing.
    @test (res == ref_vec(elt, vectort_elt)) ||
      (res == ref_vec(elt, vectort_elt, "BlockSparseArrays."))

    a = BlockSparseMatrix{elt,arrayt{elt,2}}(undef, [2, 2], [2, 2])
    res = sprint(summary, a)
    function ref_mat(elt, arrayt, prefix="")
      return "2×2-blocked 4×4 $(prefix)BlockSparseMatrix{$(elt), $(arrayt), …, …}"
    end
    # Either option is possible depending on namespacing.
    @test (res == ref_mat(elt, matrixt_elt)) ||
      (res == ref_mat(elt, matrixt_elt, "BlockSparseArrays."))

    a = BlockSparseArray{elt,3,arrayt{elt,3}}(undef, [2, 2], [2, 2], [2, 2])
    res = sprint(summary, a)
    function ref_arr(elt, arrayt, prefix="")
      return "2×2×2-blocked 4×4×4 $(prefix)BlockSparseArray{$(elt), 3, $(arrayt), …, …}"
    end
    @test (res == ref_arr(elt, arrayt_elt)) ||
      (res == ref_arr(elt, arrayt_elt, "BlockSparseArrays."))

    if elt === Float64
      # Not testing other element types since they change the
      # spacing so it isn't easy to make the test general.

      a′ = BlockSparseVector{elt,arrayt{elt,1}}(undef, [2, 2])
      @allowscalar a′[1] = 1
      a = a′
      @test sprint(show, "text/plain", a) ==
        "$(summary(a)):\n $(eltype(a)(1))\n $(zero(eltype(a)))\n ───\n  ⋅ \n  ⋅ "
      a = @view a′[:]
      @test sprint(show, "text/plain", a) ==
        "$(summary(a)):\n $(eltype(a)(1))\n $(zero(eltype(a)))\n  ⋅ \n  ⋅ "

      a′ = BlockSparseMatrix{elt,arrayt{elt,2}}(undef, [2, 2], [2, 2])
      @allowscalar a′[1, 2] = 12
      for a in (a′, @view(a′[:, :]))
        @test sprint(show, "text/plain", a) ==
          "$(summary(a)):\n $(zero(eltype(a)))  $(eltype(a)(12))  │   ⋅    ⋅ \n $(zero(eltype(a)))   $(zero(eltype(a)))  │   ⋅    ⋅ \n ───────────┼──────────\n  ⋅     ⋅   │   ⋅    ⋅ \n  ⋅     ⋅   │   ⋅    ⋅ "
      end

      a′ = BlockSparseArray{elt,3,arrayt{elt,3}}(undef, [2, 2], [2, 2], [2, 2])
      @allowscalar a′[1, 2, 1] = 121
      for a in (a′, @view(a′[:, :, :]))
        @test sprint(show, "text/plain", a) ==
          "$(summary(a)):\n[:, :, 1] =\n $(zero(eltype(a)))  $(eltype(a)(121))   ⋅    ⋅ \n $(zero(eltype(a)))    $(zero(eltype(a)))   ⋅    ⋅ \n  ⋅      ⋅    ⋅    ⋅ \n  ⋅      ⋅    ⋅    ⋅ \n\n[:, :, 2] =\n $(zero(eltype(a)))  $(zero(eltype(a)))   ⋅    ⋅ \n $(zero(eltype(a)))  $(zero(eltype(a)))   ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n\n[:, :, 3] =\n  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n\n[:, :, 4] =\n  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ "
      end
    end
  end
  @testset "TypeParameterAccessors.position" begin
    @test TypeParameterAccessors.position(BlockSparseArray, eltype) == Position(1)
    @test TypeParameterAccessors.position(BlockSparseArray, ndims) == Position(2)
    @test TypeParameterAccessors.position(BlockSparseArray, blocktype) == Position(3)
    @test TypeParameterAccessors.position(BlockSparseArray, blockstype) == Position(4)
  end
end
