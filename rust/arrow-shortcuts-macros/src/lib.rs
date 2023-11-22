use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token::{As, Bracket, Comma, Paren},
    ExprLit,
};

// Convert a type "path" (e.g. u32 or my_crate::foo::my_udt) to an arrow type
// using ArrowTypeInfer
fn path_to_arrow_type(type_path: &syn::TypePath) -> proc_macro2::TokenStream {
    quote! { <#type_path as ::arrow_shortcuts::util::ArrowTypeInfer>::arrow_type() }
}

// Convert an array type (e.g. [u32; 32]) to an arrow type
fn type_array_to_arrow_type(
    type_array: &syn::TypeArray,
    nullable: bool,
) -> proc_macro2::TokenStream {
    let inner_type = rust_type_to_arrow_type(type_array.elem.as_ref(), nullable);
    let len = &type_array.len;
    quote! {
      ::arrow_shortcuts::arrow_schema::DataType::FixedSizeList(
        ::std::sync::Arc::new(::arrow_shortcuts::arrow_schema::Field::new(
            "item",
            #inner_type,
            #nullable,
        )),
        #len,
      )
    }
}

// Convert a slice type (e.g. &[u32]) to an arrow type
fn type_slice_to_arrow_type(
    type_slice: &syn::TypeSlice,
    nullable: bool,
) -> proc_macro2::TokenStream {
    let inner_type = rust_type_to_arrow_type(type_slice.elem.as_ref(), nullable);
    quote! {
        ::arrow_shortcuts::arrow_schema::DataType::List(
            ::std::sync::Arc::new(::arrow_shortcuts::arrow_schema::Field::new(
                "item",
                #inner_type,
                #nullable
            ))
        )
    }
}

// Convert a reference type (e.g. &u32) to an arrow type.  We just treat it the same as the non-reference type
// This is mainly here for things like &str
fn type_ref_to_arrow_type(type_ref: &syn::TypeReference) -> proc_macro2::TokenStream {
    quote! { <#type_ref as ::arrow_shortcuts::util::ArrowTypeInfer>::arrow_type() }
}

// Parse the special "never" type (!) to the null arrow type
fn never_to_arrow_type() -> proc_macro2::TokenStream {
    quote! { ::arrow_shortcuts::arrow_schema::DataType::Null }
}

// Convert a rust type to an arrow type
fn rust_type_to_arrow_type(type_: &syn::Type, nullable: bool) -> proc_macro2::TokenStream {
    match type_ {
        syn::Type::Array(type_array) => type_array_to_arrow_type(type_array, nullable),
        syn::Type::BareFn(..) => panic!("The arr_type! macro is not compatible with functions"),
        syn::Type::Never(..) => never_to_arrow_type(),
        syn::Type::Paren(paren_type) => rust_type_to_arrow_type(paren_type.elem.as_ref(), false),
        syn::Type::Path(type_path) => path_to_arrow_type(type_path),
        syn::Type::Reference(type_ref) => type_ref_to_arrow_type(type_ref),
        syn::Type::Slice(type_slice) => type_slice_to_arrow_type(type_slice, nullable),
        syn::Type::Tuple(..) => panic!("union support not available in arr_type! macro"),
        _ => panic!("input cannot be converted to an arrow type"),
    }
}

// Convert rust code to an arrow field
//
// Example Input:
//  my_field: f32
//
// Example Output:
//  Field::new("my_field", DataType::Float32, true)
fn rust_field_to_arrow_field(field: &Field, nullable: bool) -> proc_macro2::TokenStream {
    let name = field.name.to_string();
    let type_ = rust_to_arrow_type(&field.ty, nullable);
    quote! {::std::sync::Arc::new(
        ::arrow_shortcuts::arrow_schema::Field::new(#name, #type_, #nullable),
    )}
}

// Convert a rust struct (which, itself, is special syntax) to an arrow struct type
fn rust_struct_to_arrow_type(struct_: &NestedType, nullable: bool) -> proc_macro2::TokenStream {
    let parsed_fields = struct_
        .fields
        .iter()
        .map(|field| rust_field_to_arrow_field(field, nullable))
        .collect::<Vec<_>>();
    quote! {
        ::arrow_shortcuts::arrow_schema::DataType::Struct(
            [#(#parsed_fields),*]
            .into(),
        )
    }
}

// Convert rust code to an arrow type
//
// Example input:
//  u32
//
// Equivalent output:
//  DataType::UInt32
//
// Actual output:
//  <u32 as ArrowTypeInfer>::arrow_type()
fn rust_to_arrow_type(type_: &ArrowTypeParam, nullable: bool) -> proc_macro2::TokenStream {
    match type_ {
        ArrowTypeParam::SimpleType(type_) => rust_type_to_arrow_type(type_, nullable),
        ArrowTypeParam::StructType(struct_) => rust_struct_to_arrow_type(struct_, nullable),
    }
}

// Convert rust code to an arrow schema
//
// Example input:
//  {
//     vector: [f32; 128],
//     metadata: {
//       caption: &str,
//       user_score: f64
//     }
//  }
//
// Example output:
//
// Schema::new([field_one, field_two])
fn rust_to_arrow_schema(schema: &NestedType) -> proc_macro2::TokenStream {
    let parsed_fields = schema
        .fields
        .iter()
        .map(|field| rust_field_to_arrow_field(field, true))
        .collect::<Vec<_>>();
    quote! {
        ::arrow_shortcuts::arrow_schema::Schema::new([#(#parsed_fields),*])
    }
}

// Convert a rust literal to an optional array item that can be fed to an arrow array constructor
//
// Example Input (not null):
//  7
// Example Output:
//  Some(7)
//
// Example Input (null):
//  ()
// Example Output (null):
//  None
fn rust_to_array_item(values: &ArrayElement) -> proc_macro2::TokenStream {
    match values {
        ArrayElement::Lit(lit) => {
            let val = &lit.lit;
            quote! { Some(#val) }
        }
        // TODO
        // ArrayElement::Array(array) => todo!(),
        ArrayElement::Null(_) => quote! { None },
    }
}

// Convert a rust type literal to an Arrow array type
//
// Example Input:
//  u32
//
// Equivalent Output:
//  UInt32Array
//
// Actual Output:
//  <u32 as ArrowTypeInfer>::ArrayType
fn path_to_array_type(type_path: &syn::TypePath) -> proc_macro2::TokenStream {
    quote! { <#type_path as ::arrow_shortcuts::util::ArrowTypeInfer>::ArrayType }
}

// Same as path_to_array_type but for ref types (e.g. &u32)
fn ref_to_array_type(type_ref: &syn::TypeReference) -> proc_macro2::TokenStream {
    quote! { <#type_ref as ::arrow_shortcuts::util::ArrowTypeInfer>::ArrayType }
}

enum TypeKind {
    Null,
    Scalar,
    // TODO
    // Array,
}

// Convert a rust type to an Arrow array type
fn type_to_array_type(type_: &syn::Type) -> (proc_macro2::TokenStream, TypeKind) {
    match type_ {
        syn::Type::Path(type_path) => (path_to_array_type(type_path), TypeKind::Scalar),
        syn::Type::Reference(type_ref) => (ref_to_array_type(type_ref), TypeKind::Scalar),
        syn::Type::Never(_) => (
            quote! { ::arrow_shortcuts::arrow_array::NullArray },
            TypeKind::Null,
        ),
        // TODO: syn::Type::Array & syn::Type::Slice should convert to FixedArray / Array TypeKind
        _ => panic!("input cannot be converted to an array type"),
    }
}

// Convert rust code to an arrow array
//
// Example Input:
//  [0, 3, ()] as u32
//
// Equivalent Output:
//  UInt32Array::from(vec![Some(0), Some(3), None])
//
// Actual Output:
//  <u32 as ArrowTypeInfer>::ArrayType::from(vec![Some(0), Some(3), None])
fn rust_to_array(array: &TypedArrowArray) -> (proc_macro2::TokenStream, u32) {
    let (array_type, type_kind) = type_to_array_type(&array.ty);
    // TODO: Add a case for array types (e.g. arr_array!([[1, 2], [3, 4]] as [u32; 2]))
    let values = array
        .array
        .elems
        .iter()
        .map(rust_to_array_item)
        .collect::<Vec<_>>();
    let num_values = values.len();
    let tokens = if matches!(type_kind, TypeKind::Null) {
        quote! { <#array_type>::new(#num_values) }
    } else {
        quote! {
            <#array_type>::from(vec![#(#values),*])
        }
    };
    (tokens, num_values as u32)
}

// Wraps an array in an Arc
fn wrap_array(arr: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    quote! {
        ::std::sync::Arc::new(#arr) as ::std::sync::Arc<dyn ::arrow_shortcuts::arrow_array::Array>
    }
}

// Convert rust code to an arrow record batch
//
// Example Input:
//  {
//    "x": [1, 2, ()] as u32,
//    "y": [0, (), 3] as f32,
//  }
//
// Example Output:
//   RecordBatch::try_new(schema, vec![x_array, y_array]).unwrap()
//
// Safety:
//   RecordBatch::try_new will fail if:
//     * there are no columns (this will fail to compile)
//     * the schema has a different length than the arrays
//         (this is not possible since both are generated from the same ground truth)
//     * the arrays have differing lengths (this will fail to compile)
//
//   All cases are accounted for and so it is safe to unwrap
fn rust_to_arrow_batch(batch: &ArrowBatch) -> syn::Result<proc_macro2::TokenStream> {
    let fields = batch
        .fields
        .iter()
        .map(|field| {
            let name = &field.name;
            let ty = &field.array.ty;
            quote! {#name: #ty}
        })
        .collect::<Vec<_>>();
    let schema = quote! {
        ::arrow_shortcuts_macros::arr_schema!({
            #(#fields),*
        })
    };
    let schema = quote! { ::std::sync::Arc::new(#schema) };
    let arrs_and_lens = batch.fields.iter().map(|field| rust_to_array(&field.array));
    let lens = arrs_and_lens
        .clone()
        .map(|(_, len)| len)
        .collect::<Vec<_>>();
    if lens.is_empty() {
        return Err(syn::Error::new(
            batch.brace_token.span.open(),
            "the arr_batch macro requires at least one array",
        ));
    }
    let first_len = lens[0];
    if let Some(mismatched_len) = lens[1..].iter().find(|&l| *l != first_len) {
        return Err(syn::Error::new(
            batch.brace_token.span.open(),
            format!("all arrays must have equal length.  First array has length {}, a subsequent array has length {}", first_len, mismatched_len)
        ));
    }
    let arrs = arrs_and_lens.map(|(arr, _)| wrap_array(arr));
    let arrs = quote! {vec![#(#arrs),*]};
    Ok(quote! {
        ::arrow_shortcuts::arrow_array::RecordBatch::try_new(#schema, #arrs).unwrap()
    })
}

// New rust syntax for a field (identifier: type)
//
// Examples:
//  foo: u32
//  blah: my_crate::MyUdf
struct Field {
    name: syn::Ident,
    _colon_token: syn::Token![:],
    ty: ArrowTypeParam,
}

impl Parse for Field {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self {
            name: input.parse()?,
            _colon_token: input.parse()?,
            ty: input.parse()?,
        })
    }
}

// New rust syntax for a nested type ({field, field, field})
//
// Examples:
//  {x: u32, y: f32, udf: my_crate::MyUdf}
//  {score: f32, points: {x: f32, y: f32}}
//  {vector: [f32; 768], label: &str}
struct NestedType {
    _brace_token: syn::token::Brace,
    fields: syn::punctuated::Punctuated<Field, syn::Token![,]>,
}

impl Parse for NestedType {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        Ok(Self {
            _brace_token: syn::braced!(content in input),
            fields: content.parse_terminated(Field::parse, syn::Token![,])?,
        })
    }
}

// A field's type, can either be a simple type or a nested type
enum ArrowTypeParam {
    SimpleType(syn::Type),
    StructType(NestedType),
}

impl syn::parse::Parse for ArrowTypeParam {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(syn::token::Brace) {
            input.parse().map(ArrowTypeParam::StructType)
        } else {
            input.parse().map(ArrowTypeParam::SimpleType)
        }
    }
}

fn arrow_type2(input: proc_macro2::TokenStream) -> syn::Result<proc_macro2::TokenStream> {
    let type_: ArrowTypeParam = syn::parse2(input)?;
    Ok(rust_to_arrow_type(&type_, true))
}

/// A macro to create arrow types from rust types
///
/// This macro takes in a rust type and desugars it into an equivalent Arrow DataType.  This
/// macro works by relying on the ArrowTypeInfer trait.  To use this macro with your own user
/// defined types you should implement the ArrowTypeInfer trait.
///
/// # Examples
/// ```ignore
/// // Basic type
/// assert_eq!(arr_type!(u32), DataType::UInt32);
/// // Fixed size list
/// assert_eq!(
/// arr_type!([i32; 5]),
///   DataType::FixedSizeList(Arc::new(Field::new("item", arr_type!(i32), true)), 5)
/// );
/// // Variable size list
/// assert_eq!(
/// arr_type!([i32]),
///   DataType::List(Arc::new(Field::new("item", arr_type!(i32), true)))
/// );
/// // Nested (struct) types
/// assert_eq!(
///   arr_type!({
///     foo: i32,
///     bar: f32,
///   }),
///   DataType::Struct([arr_field!(foo: i32), arr_field!(bar: f32),].into())
/// );
/// ```
#[proc_macro]
pub fn arr_type(input: TokenStream) -> TokenStream {
    let input = proc_macro2::TokenStream::from(input);
    arrow_type2(input).unwrap().into()
}

fn arrow_field2(input: proc_macro2::TokenStream) -> syn::Result<proc_macro2::TokenStream> {
    let field: Field = syn::parse2(input)?;
    Ok(rust_field_to_arrow_field(&field, true))
}

/// A macro to create arrow types from a field-like rust syntax
///
/// # Examples
/// ```ignore
/// assert_eq!(arr_field!(score: f32), Field::new("score", DataType::Float32, true))
/// ```
#[proc_macro]
pub fn arr_field(input: TokenStream) -> TokenStream {
    let input = proc_macro2::TokenStream::from(input);
    arrow_field2(input).unwrap().into()
}

fn arrow_schema2(input: proc_macro2::TokenStream) -> syn::Result<proc_macro2::TokenStream> {
    let struct_type: NestedType = syn::parse2(input)?;
    Ok(rust_to_arrow_schema(&struct_type))
}

/// A macro to create arrow schemas from a dictionary-like rust syntax
///
/// # Examples
/// ```ignore
/// let schema = arr_schema!({
///   vector: [f32; 128],
///   metadata: {
///     caption: &str,
///     user_score: f64
///  }
/// });
/// ```
#[proc_macro]
pub fn arr_schema(input: TokenStream) -> TokenStream {
    let input = proc_macro2::TokenStream::from(input);
    arrow_schema2(input).unwrap().into()
}

// This is just () which we interpret as null in arr_array construction
struct NullLit {
    pub _paren: Paren,
}

impl Parse for NullLit {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let _content;
        Ok(Self {
            _paren: syn::parenthesized!(_content in input),
        })
    }
}

// Syntax for an element of an array.  Either a literal or ()
enum ArrayElement {
    Null(NullLit),
    Lit(ExprLit),
    // TODO: support arrays of arrays
    // Array(ArrowArray),
}

impl Parse for ArrayElement {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(syn::token::Paren) {
            input.parse().map(ArrayElement::Null)
        } else if lookahead.peek(syn::token::Bracket) {
            todo!()
            // input.parse().map(ArrayElement::Array)
        } else {
            input.parse().map(ArrayElement::Lit)
        }
    }
}

// Syntax for an array ([value, value, value])
struct ArrowArray {
    pub _bracket: Bracket,
    pub elems: Punctuated<ArrayElement, Comma>,
}

// Syntax for an array with a type ([value, value, value] as type)
//
// Examples:
//  [0, (), 5] as u32
//  [(), ()] as f64
//  [(), ()] as !
struct TypedArrowArray {
    pub array: ArrowArray,
    pub _as: As,
    pub ty: syn::Type,
}

impl Parse for ArrowArray {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        Ok(Self {
            _bracket: syn::bracketed!(content in input),
            elems: content.parse_terminated(ArrayElement::parse, syn::Token![,])?,
        })
    }
}

impl Parse for TypedArrowArray {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self {
            array: input.parse::<ArrowArray>()?,
            _as: input.parse::<As>()?,
            ty: input.parse::<syn::Type>()?,
        })
    }
}

fn arrow_array2(input: proc_macro2::TokenStream) -> syn::Result<proc_macro2::TokenStream> {
    let arrow_array: TypedArrowArray = syn::parse2(input)?;
    Ok(rust_to_array(&arrow_array).0)
}

/// A macro to create arrow arrays from an array-like rust syntax
///
/// # Examples
/// ```ignore
/// assert_eq!(
///     arr_array!([1, (), 5] as u8),
///     UInt8Array::from(vec![Some(1), None, Some(5)])
/// );
/// assert_eq!(arr_array!([(), ()] as !), NullArray::new(2),)
/// ```
#[proc_macro]
pub fn arr_array(input: TokenStream) -> TokenStream {
    let input = proc_macro2::TokenStream::from(input);
    arrow_array2(input).unwrap().into()
}

// Syntax for a column in a record batch (name: [value, value] as type)
struct BatchField {
    name: syn::Ident,
    _colon_token: syn::Token![:],
    array: TypedArrowArray,
}

impl Parse for BatchField {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self {
            name: input.parse()?,
            _colon_token: input.parse()?,
            array: input.parse()?,
        })
    }
}

// Syntax for a record batch ({batch_field, batch_field, batch_field})
struct ArrowBatch {
    brace_token: syn::token::Brace,
    fields: syn::punctuated::Punctuated<BatchField, syn::Token![,]>,
}

impl Parse for ArrowBatch {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        Ok(Self {
            brace_token: syn::braced!(content in input),
            fields: content.parse_terminated(BatchField::parse, syn::Token![,])?,
        })
    }
}

fn arrow_batch2(input: proc_macro2::TokenStream) -> syn::Result<proc_macro2::TokenStream> {
    let arrow_batch: ArrowBatch = syn::parse2(input)?;
    rust_to_arrow_batch(&arrow_batch)
}

/// A macro to create arrow arrays from an array-like rust syntax
///
/// # Examples
/// ```ignore
/// let batch = arr_batch!({
///   x: [1, 2, ()] as u8,
///   y: [4, (), 5] as u16,
///   strings: [(), "x", "y"] as &str,
///   // Not yet supported
///   // vecs: [[1, 2, 3], [4, (), 6], ()] as [u16; 3],
/// });
/// ```
#[proc_macro]
pub fn arr_batch(input: TokenStream) -> TokenStream {
    let input = proc_macro2::TokenStream::from(input);
    arrow_batch2(input).unwrap().into()
}
