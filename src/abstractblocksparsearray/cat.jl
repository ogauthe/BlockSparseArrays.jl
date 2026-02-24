using DerivableInterfaces.Concatenate: concatenate
using DerivableInterfaces: @interface, interface

function Base._cat(dims, as::AnyAbstractBlockSparseArray...)
    return concatenate(dims, as...)
end
