Primitive Attributes: accumulation mode {#dev_guide_attributes_accumulation_mode}
===================================================================

Some applications can benefit from using lower precision accumulators
to speed up computations without causing noticeable impacts on
accuracy.

The default numerical behavior of oneDNN (described in @ref
dev_guide_data_types) can be altered to allow the use of low precision
accumulators. When passed to a primitive creation, the @ref
dnnl::accumulation_mode primitive attribute specifies which datatype
can be used for accumulation purposes for that given primitive.

The @ref dnnl::accumulation_mode primitive attribute accepts:
- `strict` (default): For floating-point primitives (as defined in
@ref dev_guide_data_types), the default accumulation datatype is f32
(or f64 for f64 primitives).  For integral primitives (as defined in
@ref dev_guide_data_types), the default accumulation datatype is s32.
- `relaxed`: Same as strict except some partial accumulators can be
  rounded to the src/dst datatype in memory.
- `any`: Uses the fastest implementation available with one of the
  src/dst datatypes or a higher precision accumulation datatype.
- `f32`, `f16` and `s32`: Uses the specified accumulation datatype.
