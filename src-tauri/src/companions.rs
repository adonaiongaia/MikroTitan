//! Smart model companions: GGUF metadata inspection, vision-projector and
//! speculative-draft pairing, and saved model bundles.
//!
//! Everything here reads only the GGUF header, the key/value metadata block and
//! the tensor directory. Tensor payloads are never loaded, so inspecting a 60 GB
//! file costs the same as inspecting a 200 MB one: a few hundred kilobytes of
//! header reads.
//!
//! Two pairing rules are deliberate and non-negotiable:
//!
//! * A speculative draft model must share the main model's `general.architecture`
//!   **and** its exact vocabulary size. llama-server aborts on a mismatch.
//! * A vision projector is only auto-selected when it belongs to the same vision
//!   family, or sits beside the main model and no family can be read. Guessing
//!   across families is what makes llama-server exit at load time.

use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
    time::UNIX_EPOCH,
};

use serde::{Deserialize, Serialize};
use tauri::State;

use crate::{
    error::{AppError, AppResult},
    settings::{expand_home, AppSettings, SettingsStore},
    util::now_ms,
};

// ---------------------------------------------------------------------------
// GGUF format constants
// ---------------------------------------------------------------------------

const TYPE_UINT8: u32 = 0;
const TYPE_INT8: u32 = 1;
const TYPE_UINT16: u32 = 2;
const TYPE_INT16: u32 = 3;
const TYPE_UINT32: u32 = 4;
const TYPE_INT32: u32 = 5;
const TYPE_FLOAT32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_UINT64: u32 = 10;
const TYPE_INT64: u32 = 11;
const TYPE_FLOAT64: u32 = 12;

const MAX_METADATA_ENTRIES: u64 = 1_000_000;
const MAX_KEY_BYTES: u64 = 16 * 1024;
const MAX_STRING_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ARRAY_ENTRIES: u64 = 16_000_000;
const MAX_ARRAY_DEPTH: u8 = 8;
const MAX_TENSOR_COUNT: u64 = 200_000;
const MAX_TENSOR_NAME_BYTES: u64 = 4 * 1024;
const MAX_TENSOR_DIMS: u32 = 8;

const DRAFT_PARAMETER_LIMIT: u64 = 3_000_000_000;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// What a `.gguf` file is, as far as the studio can prove from its header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelRole {
    /// A standard text LLM.
    MainLlm,
    /// A multimodal LLM that expects a separate projector before it accepts images.
    VisionMain,
    /// A CLIP/SigLIP vision projector (`--mmproj`).
    Mmproj,
    /// A small same-family model that can draft tokens for a larger main model.
    DraftCandidate,
}

impl ModelRole {
    pub fn is_main(self) -> bool {
        matches!(self, ModelRole::MainLlm | ModelRole::VisionMain)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GgufCompanionInfo {
    pub path: String,
    pub file_name: String,
    pub role: ModelRole,
    /// `general.architecture`, e.g. `qwen2`, `llama`, `gemma3`, `clip`.
    pub architecture: Option<String>,
    /// `general.name`, when the converter wrote one.
    pub name: Option<String>,
    /// `tokenizer.ggml.model`, e.g. `gpt2`, `llama`, `spm`.
    pub tokenizer_model: Option<String>,
    /// Exact vocabulary size: the token array count, else the embedding rows.
    pub vocab_size: Option<u32>,
    pub context_length: Option<u64>,
    /// Sum of every tensor's element count, read from the tensor directory.
    pub parameter_estimate: Option<u64>,
    pub byte_size: u64,
    pub quant: String,
    pub tensor_count: u32,
    /// How many tensors look like a vision tower (`mm.*`, `v.*`, `vision.*`).
    pub vision_tensor_count: u32,
    /// Metadata keys that identify a multimodal file.
    pub vision_markers: Vec<String>,
    /// Normalised family used for pairing (`qwen2.5-vl`, `gemma-3`, `llava`, …).
    pub vision_family: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompanionCandidate {
    pub path: String,
    pub file_name: String,
    pub architecture: Option<String>,
    pub vocab_size: Option<u32>,
    pub quant: String,
    pub parameter_estimate: Option<u64>,
    pub size_bytes: u64,
    pub score: i32,
    pub compatible: bool,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct VisionPairing {
    /// A projector that is currently bound in settings and still valid.
    pub bound_path: Option<String>,
    pub recommended: Option<CompanionCandidate>,
    /// Every local projector, best first. `compatible` says whether it may be
    /// paired with this main model without a family mismatch.
    pub candidates: Vec<CompanionCandidate>,
    /// True when the main model needs a projector but none is bound.
    pub missing: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct DraftPairing {
    pub bound_path: Option<String>,
    pub recommended: Option<CompanionCandidate>,
    /// Every local draft-sized model, with an exact vocab/architecture check.
    pub candidates: Vec<CompanionCandidate>,
    /// True when a draft is bound.
    pub paired: bool,
    pub message: String,
    /// Bytes of the main model plus the best compatible draft.
    pub combined_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompanionPairing {
    pub main: GgufCompanionInfo,
    pub vision: VisionPairing,
    pub draft: DraftPairing,
    /// Main + recommended projector + recommended draft.
    pub combined_size_bytes: u64,
    /// The same stack with llama.cpp's runtime and KV-cache headroom.
    pub estimated_runtime_bytes: u64,
    /// Resident system-RAM estimate for the full stack.
    pub estimated_ram_bytes: u64,
    /// VRAM estimate when every part is offloaded to the GPU.
    pub estimated_vram_bytes: u64,
    /// The download repository, when the library manifest knows it.
    pub repo: Option<String>,
    pub active_bundle_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct LibraryCompanions {
    /// Every inspected GGUF, including projectors and draft models.
    pub models: Vec<GgufCompanionInfo>,
    /// Pairing verdicts for the main models only.
    pub pairings: Vec<CompanionPairing>,
    pub warnings: Vec<String>,
}

/// A `(main, mmproj, draft, draft_max)` tuple persisted in settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelBundle {
    pub id: String,
    pub name: String,
    pub main_model_path: String,
    pub mmproj_path: String,
    pub draft_model_path: String,
    pub draft_max_tokens: u32,
    pub draft_min_tokens: u32,
    pub draft_p_min: f32,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

impl Default for ModelBundle {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            main_model_path: String::new(),
            mmproj_path: String::new(),
            draft_model_path: String::new(),
            draft_max_tokens: 8,
            draft_min_tokens: 0,
            draft_p_min: 0.75,
            created_at_ms: 0,
            updated_at_ms: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct BundleCheck {
    pub ok: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub main: Option<GgufCompanionInfo>,
    pub mmproj: Option<GgufCompanionInfo>,
    pub draft: Option<GgufCompanionInfo>,
    pub combined_size_bytes: u64,
    pub estimated_runtime_bytes: u64,
}

// ---------------------------------------------------------------------------
// GGUF reader
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct ParsedHeader {
    architecture: Option<String>,
    name: Option<String>,
    tokenizer_model: Option<String>,
    vocab_size: Option<u32>,
    context_length: Option<u64>,
    vision_markers: Vec<String>,
    clip_architecture: bool,
    tensor_count: u32,
    vision_tensor_count: u32,
    parameter_estimate: u64,
    type_counts: BTreeMap<u32, u64>,
    token_embd_dims: Option<Vec<u64>>,
}

fn read_gguf<R: Read + Seek>(reader: &mut R) -> Result<ParsedHeader, String> {
    let mut magic = [0u8; 4];
    read_exact(reader, &mut magic)?;
    if &magic != b"GGUF" {
        return Err("this is not a GGUF file (missing GGUF header)".into());
    }

    let version = read_u32(reader)?;
    if !(2..=3).contains(&version) {
        return Err(format!(
            "unsupported GGUF version {version}; this build reads version 2 and 3 files"
        ));
    }
    let tensor_count = read_u64(reader)?;
    let metadata_count = read_u64(reader)?;
    if tensor_count > MAX_TENSOR_COUNT {
        return Err(format!(
            "tensor directory declares an unreasonable {tensor_count} tensors"
        ));
    }
    if metadata_count > MAX_METADATA_ENTRIES {
        return Err(format!(
            "metadata declares an unreasonable {metadata_count} entries"
        ));
    }

    let mut parsed = ParsedHeader::default();

    for _ in 0..metadata_count {
        let key_bytes = read_sized(reader, MAX_KEY_BYTES, "metadata key")?;
        let key = String::from_utf8_lossy(&key_bytes).to_string();
        let lower = key.to_ascii_lowercase();
        let value_type = read_u32(reader)?;

        if parsed.vision_markers.len() < 16
            && (lower.starts_with("clip.")
                || lower.contains("mmproj")
                || lower.contains("vision")
                || lower.contains("multimodal"))
        {
            parsed.vision_markers.push(key.clone());
        }

        if lower == "general.architecture" {
            parsed.architecture = read_string_value(reader, value_type, MAX_KEY_BYTES)?;
            if parsed
                .architecture
                .as_deref()
                .map(|arch| arch.eq_ignore_ascii_case("clip"))
                .unwrap_or(false)
            {
                parsed.clip_architecture = true;
            }
            continue;
        }

        if lower == "general.name" {
            parsed.name = read_string_value(reader, value_type, MAX_KEY_BYTES)?;
            continue;
        }

        if lower == "tokenizer.ggml.model" {
            parsed.tokenizer_model = read_string_value(reader, value_type, MAX_KEY_BYTES)?;
            continue;
        }

        // `<architecture>.context_length` and `general.context_length`.
        if lower.ends_with(".context_length") {
            if let Some(value) = read_unsigned_value(reader, value_type)? {
                if value > 0 {
                    parsed.context_length = Some(value);
                }
            }
            continue;
        }

        // The token array can hold hundreds of thousands of strings. Record its
        // count, then skip the elements without allocating them.
        if lower == "tokenizer.ggml.tokens" {
            if value_type == TYPE_ARRAY {
                let inner_type = read_u32(reader)?;
                let count = read_u64(reader)?;
                if count > MAX_ARRAY_ENTRIES {
                    return Err(format!(
                        "tokenizer.ggml.tokens declares an unreasonable {count} entries"
                    ));
                }
                if parsed.vocab_size.is_none() && count > 0 && count <= u32::MAX as u64 {
                    parsed.vocab_size = Some(count as u32);
                }
                skip_array_items(reader, inner_type, count, 1)?;
            } else {
                skip_value(reader, value_type, 0)?;
            }
            continue;
        }

        skip_value(reader, value_type, 0)?;
    }

    // Read the tensor directory: name, dimensions, ggml type and offset. This
    // gives an exact parameter count and real evidence of a vision tower
    // without touching a single weight.
    for _ in 0..tensor_count {
        let name_bytes = read_sized(reader, MAX_TENSOR_NAME_BYTES, "tensor name")?;
        let name = String::from_utf8_lossy(&name_bytes).to_string();
        let lower_name = name.to_ascii_lowercase();

        let dims_count = read_u32(reader)?;
        if dims_count > MAX_TENSOR_DIMS {
            return Err(format!(
                "tensor {name} declares an unreasonable {dims_count} dimensions"
            ));
        }
        let mut dims = Vec::with_capacity(dims_count as usize);
        let mut elements: u64 = 1;
        for _ in 0..dims_count {
            let dim = read_u64(reader)?;
            dims.push(dim);
            elements = elements.saturating_mul(dim.max(1));
        }
        let ggml_type = read_u32(reader)?;
        let _offset = read_u64(reader)?;

        parsed.tensor_count = parsed.tensor_count.saturating_add(1);
        parsed.parameter_estimate = parsed.parameter_estimate.saturating_add(elements);
        *parsed.type_counts.entry(ggml_type).or_insert(0) += elements;

        if parsed.token_embd_dims.is_none()
            && (lower_name.contains("token_embd")
                || lower_name.contains("tok_embeddings")
                || lower_name.contains("word_embeddings"))
            && dims.len() >= 2
        {
            parsed.token_embd_dims = Some(dims.clone());
        }

        if parsed.vision_tensor_count < 64_000 && is_vision_tensor(&lower_name) {
            parsed.vision_tensor_count += 1;
            if parsed.vision_markers.len() < 16 {
                parsed.vision_markers.push(name);
            }
        }
    }

    // The tensor directory is authoritative for the vocabulary when the
    // tokenizer array was absent (some projectors and pruned files).
    if parsed.vocab_size.is_none() {
        if let Some(dims) = parsed.token_embd_dims.as_ref() {
            if let Some(vocab) = dims.get(1).copied().filter(|value| *value > 0) {
                if vocab <= u32::MAX as u64 {
                    parsed.vocab_size = Some(vocab as u32);
                }
            }
        }
    }

    Ok(parsed)
}

fn is_vision_tensor(name: &str) -> bool {
    name.starts_with("mm.")
        || name.starts_with("v.")
        || name.starts_with("vision")
        || name.starts_with("v_blk")
        || name.starts_with("v_")
        || name.contains("mmproj")
}

fn read_string_value<R: Read + Seek>(
    reader: &mut R,
    value_type: u32,
    max: u64,
) -> Result<Option<String>, String> {
    if value_type != TYPE_STRING {
        skip_value(reader, value_type, 0)?;
        return Ok(None);
    }
    let length = read_u64(reader)?;
    if length > max {
        skip_bytes(reader, length)?;
        return Ok(None);
    }
    let length = usize::try_from(length).map_err(|_| "metadata string is too large".to_string())?;
    let mut bytes = vec![0u8; length];
    read_exact(reader, &mut bytes)?;
    Ok(Some(String::from_utf8_lossy(&bytes).trim().to_string()))
}

fn read_unsigned_value<R: Read + Seek>(
    reader: &mut R,
    value_type: u32,
) -> Result<Option<u64>, String> {
    match value_type {
        TYPE_UINT8 => Ok(Some(read_u8(reader)? as u64)),
        TYPE_INT8 => {
            let value = read_u8(reader)? as i8;
            Ok((value >= 0).then_some(value as u64))
        }
        TYPE_UINT16 => Ok(Some(read_u16(reader)? as u64)),
        TYPE_INT16 => {
            let value = read_u16(reader)? as i16;
            Ok((value >= 0).then_some(value as u64))
        }
        TYPE_UINT32 => Ok(Some(read_u32(reader)? as u64)),
        TYPE_INT32 => {
            let value = read_u32(reader)? as i32;
            Ok((value >= 0).then_some(value as u64))
        }
        TYPE_UINT64 => Ok(Some(read_u64(reader)?)),
        TYPE_INT64 => {
            let value = read_u64(reader)? as i64;
            Ok((value >= 0).then_some(value as u64))
        }
        TYPE_FLOAT32 => {
            let _ = read_u32(reader)?;
            Ok(None)
        }
        TYPE_FLOAT64 => {
            let _ = read_u64(reader)?;
            Ok(None)
        }
        TYPE_BOOL => {
            let _ = read_u8(reader)?;
            Ok(None)
        }
        other => {
            skip_value(reader, other, 0)?;
            Ok(None)
        }
    }
}

fn fixed_value_size(value_type: u32) -> Option<u64> {
    match value_type {
        TYPE_UINT8 | TYPE_INT8 | TYPE_BOOL => Some(1),
        TYPE_UINT16 | TYPE_INT16 => Some(2),
        TYPE_UINT32 | TYPE_INT32 | TYPE_FLOAT32 => Some(4),
        TYPE_UINT64 | TYPE_INT64 | TYPE_FLOAT64 => Some(8),
        _ => None,
    }
}

fn skip_value<R: Read + Seek>(reader: &mut R, value_type: u32, depth: u8) -> Result<(), String> {
    if depth > MAX_ARRAY_DEPTH {
        return Err("metadata arrays are nested too deeply".into());
    }
    if let Some(size) = fixed_value_size(value_type) {
        return skip_bytes(reader, size);
    }

    match value_type {
        TYPE_STRING => {
            let length = read_u64(reader)?;
            if length > MAX_STRING_BYTES {
                return Err(format!(
                    "metadata string declares an unreasonable {length} bytes"
                ));
            }
            skip_bytes(reader, length)
        }
        TYPE_ARRAY => {
            let inner_type = read_u32(reader)?;
            let count = read_u64(reader)?;
            if count > MAX_ARRAY_ENTRIES {
                return Err(format!(
                    "metadata array declares an unreasonable {count} entries"
                ));
            }
            skip_array_items(reader, inner_type, count, depth + 1)
        }
        other => Err(format!("unsupported GGUF metadata type {other}")),
    }
}

/// Skip `count` values of `inner_type` when the array header was already read.
fn skip_array_items<R: Read + Seek>(
    reader: &mut R,
    inner_type: u32,
    count: u64,
    depth: u8,
) -> Result<(), String> {
    if depth > MAX_ARRAY_DEPTH {
        return Err("metadata arrays are nested too deeply".into());
    }
    if let Some(size) = fixed_value_size(inner_type) {
        let bytes = size
            .checked_mul(count)
            .ok_or_else(|| "metadata array size overflowed".to_string())?;
        return skip_bytes(reader, bytes);
    }
    for _ in 0..count {
        skip_value(reader, inner_type, depth)?;
    }
    Ok(())
}

/// Read or seek past a run of bytes. Short skips stay in the BufReader buffer;
/// long runs seek, so a huge tokenizer or metadata blob never has to be read.
fn skip_bytes<R: Read + Seek>(reader: &mut R, length: u64) -> Result<(), String> {
    if length == 0 {
        return Ok(());
    }
    if length > 64 * 1024 {
        let offset = i64::try_from(length).map_err(|_| "metadata offset is too large".to_string())?;
        return reader
            .seek(SeekFrom::Current(offset))
            .map(|_| ())
            .map_err(|err| format!("could not skip GGUF metadata: {err}"));
    }
    let mut remaining = length;
    let mut scratch = [0u8; 8192];
    while remaining > 0 {
        let take = remaining.min(scratch.len() as u64) as usize;
        read_exact(reader, &mut scratch[..take])?;
        remaining -= take as u64;
    }
    Ok(())
}

fn read_sized<R: Read>(reader: &mut R, max: u64, label: &str) -> Result<Vec<u8>, String> {
    let length = read_u64(reader)?;
    if length > max {
        return Err(format!("{label} declares an unreasonable {length} bytes"));
    }
    let length = usize::try_from(length).map_err(|_| format!("{label} is too large"))?;
    let mut bytes = vec![0u8; length];
    read_exact(reader, &mut bytes)?;
    Ok(bytes)
}

fn read_exact<R: Read>(reader: &mut R, buffer: &mut [u8]) -> Result<(), String> {
    reader
        .read_exact(buffer)
        .map_err(|err| format!("unexpected end of GGUF metadata: {err}"))
}

fn read_u8<R: Read>(reader: &mut R) -> Result<u8, String> {
    let mut bytes = [0u8; 1];
    read_exact(reader, &mut bytes)?;
    Ok(bytes[0])
}

fn read_u16<R: Read>(reader: &mut R) -> Result<u16, String> {
    let mut bytes = [0u8; 2];
    read_exact(reader, &mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, String> {
    let mut bytes = [0u8; 4];
    read_exact(reader, &mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, String> {
    let mut bytes = [0u8; 8];
    read_exact(reader, &mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

fn ggml_type_name(value: u32) -> Option<&'static str> {
    Some(match value {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        35 => "TQ1_0",
        36 => "TQ2_0",
        39 => "MXFP4",
        _ => return None,
    })
}

const QUANT_NAMES: &[&str] = &[
    "UD-Q6_K_XL",
    "UD-Q5_K_XL",
    "UD-Q4_K_XL",
    "UD-Q3_K_XL",
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q8_0",
    "Q4_1",
    "Q4_0",
    "Q5_1",
    "Q3_K_M",
    "Q3_K_S",
    "Q3_K_L",
    "Q2_K",
    "IQ4_XS",
    "IQ4_NL",
    "IQ3_S",
    "MXFP4",
    "BF16",
    "F16",
];

fn quant_from_name(file_name: &str) -> Option<String> {
    let upper = file_name.to_ascii_uppercase();
    QUANT_NAMES
        .iter()
        .find(|token| upper.contains(**token))
        .map(|token| token.to_string())
}

/// Normalise a filename or model name to a comparable vision family.
fn detect_vision_family(text: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase().replace('_', "-");
    let compact = lower.replace(' ', "-");
    let has = |needle: &str| compact.contains(needle);

    if has("qwen3-vl") {
        return Some("qwen3-vl".into());
    }
    if has("qwen2.5-vl") || has("qwen2-5-vl") {
        return Some("qwen2.5-vl".into());
    }
    if has("qwen2-vl") {
        return Some("qwen2-vl".into());
    }
    if has("qwen-vl") || has("qwen2.5vl") {
        return Some("qwen-vl".into());
    }
    if has("minicpm-v") || has("minicpmv") {
        return Some("minicpm-v".into());
    }
    if has("gemma-3") || has("gemma3") {
        return Some("gemma-3".into());
    }
    if has("mistral-small-3") || has("mistral-small3") {
        return Some("mistral-small-3".into());
    }
    if has("pixtral") {
        return Some("pixtral".into());
    }
    if has("llava") {
        return Some("llava".into());
    }
    if has("internvl") {
        return Some("internvl".into());
    }
    if has("moondream") {
        return Some("moondream".into());
    }
    if has("phi-4-multimodal") || has("phi4-multimodal") || has("phi-4-mm") {
        return Some("phi-4-multimodal".into());
    }
    if has("smolvlm") {
        return Some("smolvlm".into());
    }
    if has("idefics") {
        return Some("idefics".into());
    }
    if has("glm-4v") || has("glm4v") {
        return Some("glm-4v".into());
    }
    if has("granite-vision") || has("granite3.2-vision") {
        return Some("granite-vision".into());
    }
    if has("ovis") {
        return Some("ovis".into());
    }
    None
}

/// A filename that asks for a projector, with the Gemma-3 text-only sizes
/// excluded (1B and 270M have no vision tower).
fn filename_expects_vision(file_name: &str, parameters: Option<u64>) -> Option<String> {
    let family = detect_vision_family(file_name)?;
    if family == "gemma-3" {
        let text_only = parameters.map(|value| value < 2_500_000_000).unwrap_or(false);
        if text_only {
            return None;
        }
    }
    Some(family)
}

fn classify(
    file_name: &str,
    architecture: Option<&str>,
    name: Option<&str>,
    vocab_size: Option<u32>,
    vision_tensor_count: u32,
    vision_markers: &[String],
    clip_architecture: bool,
    parameters: Option<u64>,
) -> (ModelRole, Option<String>) {
    let lower = file_name.to_ascii_lowercase();
    let haystack = format!(
        "{} {}",
        file_name,
        name.unwrap_or_default()
    );

    let family = detect_vision_family(&haystack);
    let is_mmproj_name = lower.contains("mmproj");
    let clip_arch = clip_architecture
        || architecture
            .map(|arch| arch.eq_ignore_ascii_case("clip"))
            .unwrap_or(false);
    let looks_like_projector = is_mmproj_name
        || clip_arch
        || (vocab_size.is_none()
            && vision_tensor_count > 0
            && vision_markers.iter().any(|marker| {
                let marker = marker.to_ascii_lowercase();
                marker.contains("clip") || marker.contains("vision")
            }));

    if looks_like_projector {
        return (ModelRole::Mmproj, family);
    }

    let metadata_says_vision = !vision_markers.is_empty() && vision_tensor_count > 0;
    let name_says_vision = filename_expects_vision(&haystack, parameters).is_some();
    if metadata_says_vision || name_says_vision {
        let family = family.or_else(|| filename_expects_vision(&haystack, parameters));
        return (ModelRole::VisionMain, family);
    }

    if let Some(parameters) = parameters {
        if parameters > 0 && parameters <= DRAFT_PARAMETER_LIMIT && vocab_size.is_some() {
            return (ModelRole::DraftCandidate, None);
        }
    }

    (ModelRole::MainLlm, None)
}

fn role_label(role: ModelRole) -> &'static str {
    match role {
        ModelRole::MainLlm => "main model",
        ModelRole::VisionMain => "vision model",
        ModelRole::Mmproj => "vision projector",
        ModelRole::DraftCandidate => "draft model",
    }
}

// ---------------------------------------------------------------------------
// Inspection
// ---------------------------------------------------------------------------

struct CachedInfo {
    len: u64,
    modified_ms: u128,
    info: GgufCompanionInfo,
}

static INFO_CACHE: OnceLock<Mutex<HashMap<PathBuf, CachedInfo>>> = OnceLock::new();

fn cache() -> &'static Mutex<HashMap<PathBuf, CachedInfo>> {
    INFO_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn modified_ms(path: &Path) -> u128 {
    fs::metadata(path)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn inspect_path(path: &Path, use_cache: bool) -> AppResult<GgufCompanionInfo> {
    let metadata = fs::metadata(path).map_err(|err| {
        AppError::message(format!(
            "Could not read {}: {err}",
            path.display()
        ))
    })?;
    if !metadata.is_file() {
        return Err(AppError::message(format!(
            "{} is not a file.",
            path.display()
        )));
    }
    let key = path.to_path_buf();
    let stamp = (metadata.len(), modified_ms(path));
    if use_cache {
        if let Ok(guard) = cache().lock() {
            if let Some(cached) = guard.get(&key) {
                if cached.len == stamp.0 && cached.modified_ms == stamp.1 {
                    return Ok(cached.info.clone());
                }
            }
        }
    }

    let file = File::open(path).map_err(|err| {
        AppError::message(format!(
            "Could not open {} to read its GGUF header: {err}",
            path.display()
        ))
    })?;
    let parsed = read_gguf(&mut BufReader::with_capacity(256 * 1024, file)).map_err(|err| {
        AppError::message(format!(
            "Could not read GGUF metadata from {}: {err}",
            path.display()
        ))
    })?;

    let file_name = path
        .file_name()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_else(|| "model.gguf".into());
    let parameter_estimate = (parsed.tensor_count > 0).then_some(parsed.parameter_estimate);
    let quant = parsed
        .type_counts
        .iter()
        .max_by_key(|(_, elements)| **elements)
        .and_then(|(value, _)| ggml_type_name(*value))
        .map(str::to_string)
        .or_else(|| quant_from_name(&file_name))
        .unwrap_or_else(|| "GGUF".into());

    let (role, vision_family) = classify(
        &file_name,
        parsed.architecture.as_deref(),
        parsed.name.as_deref(),
        parsed.vocab_size,
        parsed.vision_tensor_count,
        &parsed.vision_markers,
        parsed.clip_architecture,
        parameter_estimate,
    );

    let info = GgufCompanionInfo {
        path: path.to_string_lossy().to_string(),
        file_name,
        role,
        architecture: parsed.architecture,
        name: parsed.name,
        tokenizer_model: parsed.tokenizer_model,
        vocab_size: parsed.vocab_size,
        context_length: parsed.context_length,
        parameter_estimate,
        byte_size: metadata.len(),
        quant,
        tensor_count: parsed.tensor_count,
        vision_tensor_count: parsed.vision_tensor_count,
        vision_markers: parsed.vision_markers,
        vision_family,
    };

    if use_cache {
        if let Ok(mut guard) = cache().lock() {
            if guard.len() > 512 {
                guard.clear();
            }
            guard.insert(
                key,
                CachedInfo {
                    len: stamp.0,
                    modified_ms: stamp.1,
                    info: info.clone(),
                },
            );
        }
    }
    Ok(info)
}

pub fn inspect_gguf_metadata(path: &str) -> AppResult<GgufCompanionInfo> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(AppError::message("Choose a .gguf file to inspect."));
    }
    let expanded = expand_home(trimmed);
    if !expanded.is_file() {
        return Err(AppError::message(format!(
            "Could not inspect GGUF metadata because {} does not exist.",
            expanded.display()
        )));
    }
    inspect_path(&expanded, true)
}

// ---------------------------------------------------------------------------
// Library scanning and pairing
// ---------------------------------------------------------------------------

fn scan_gguf_files(dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
        if depth > 4 || out.len() >= 600 {
            return;
        }
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, depth + 1, out);
                continue;
            }
            if path
                .extension()
                .map(|extension| extension.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
            {
                out.push(path);
            }
        }
    }

    let mut files = Vec::new();
    walk(dir, 0, &mut files);
    files.sort_by(|a, b| a.to_string_lossy().cmp(&b.to_string_lossy()));
    files
}

fn same_path(left: &str, right: &str) -> bool {
    if left.trim().is_empty() || right.trim().is_empty() {
        return false;
    }
    let left = expand_home(left);
    let right = expand_home(right);
    if left == right {
        return true;
    }
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn same_directory(main: &GgufCompanionInfo, candidate: &GgufCompanionInfo) -> bool {
    Path::new(&main.path).parent() == Path::new(&candidate.path).parent()
}

fn family_tokens(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .split(|character: char| !character.is_ascii_alphanumeric() && character != '.')
        .filter(|token| token.len() >= 3)
        .filter(|token| {
            !matches!(
                *token,
                "gguf"
                    | "model"
                    | "mmproj"
                    | "instruct"
                    | "chat"
                    | "base"
                    | "bf16"
                    | "f16"
                    | "imatrix"
                    | "part"
            )
        })
        .map(str::to_string)
        .collect()
}

fn token_overlap(left: &str, right: &str) -> usize {
    let left = family_tokens(left);
    let right = family_tokens(right);
    left.iter().filter(|token| right.contains(token)).count()
}

fn candidate_from_info(
    info: &GgufCompanionInfo,
    score: i32,
    compatible: bool,
    reasons: Vec<String>,
) -> CompanionCandidate {
    CompanionCandidate {
        path: info.path.clone(),
        file_name: info.file_name.clone(),
        architecture: info.architecture.clone(),
        vocab_size: info.vocab_size,
        quant: info.quant.clone(),
        parameter_estimate: info.parameter_estimate,
        size_bytes: info.byte_size,
        score,
        compatible,
        reasons,
    }
}

fn score_mmproj(main: &GgufCompanionInfo, candidate: &GgufCompanionInfo) -> (i32, bool, Vec<String>) {
    let mut score = 0i32;
    let mut reasons = Vec::new();
    let mut compatible = false;

    if same_directory(main, candidate) {
        score += 30;
        reasons.push("same folder".into());
    }

    match (&main.vision_family, &candidate.vision_family) {
        (Some(left), Some(right)) if left == right => {
            score += 70;
            compatible = true;
            reasons.push(format!("same vision family ({left})"));
        }
        (Some(left), Some(right)) => {
            reasons.push(format!("different vision family ({left} vs {right})"));
        }
        _ => {}
    }

    let overlap = token_overlap(
        &format!("{} {}", main.file_name, main.name.as_deref().unwrap_or_default()),
        &format!(
            "{} {}",
            candidate.file_name,
            candidate.name.as_deref().unwrap_or_default()
        ),
    );
    if overlap > 0 {
        score += (overlap.min(4) * 8) as i32;
        reasons.push(format!("{overlap} shared name token(s)"));
    }

    if !compatible
        && same_directory(main, candidate)
        && (main.vision_family.is_none() || candidate.vision_family.is_none())
    {
        compatible = true;
        reasons.push("same folder and no conflicting family".into());
    }

    match candidate.quant.to_ascii_lowercase().as_str() {
        "f16" => {
            score += 18;
            reasons.push("f16 projector (recommended)".into());
        }
        "bf16" => {
            score += 17;
            reasons.push("bf16 projector (recommended)".into());
        }
        "q8_0" => {
            score += 12;
            reasons.push("q8_0 projector".into());
        }
        "f32" => {
            score += 10;
            reasons.push("f32 projector (largest)".into());
        }
        _ => {}
    }
    if candidate.file_name.to_ascii_lowercase().starts_with("mmproj") {
        score += 6;
    }

    (score, compatible, reasons)
}

fn draft_compatibility(
    main: &GgufCompanionInfo,
    candidate: &GgufCompanionInfo,
) -> Result<Vec<String>, String> {
    if candidate.role != ModelRole::DraftCandidate {
        return Err(format!("{} is a {}", candidate.file_name, role_label(candidate.role)));
    }
    let (Some(main_arch), Some(candidate_arch)) =
        (main.architecture.as_deref(), candidate.architecture.as_deref())
    else {
        return Err("architecture is missing from one of the files".into());
    };
    if !main_arch.eq_ignore_ascii_case(candidate_arch) {
        return Err(format!(
            "architecture differs ({main_arch} vs {candidate_arch})"
        ));
    }
    let (Some(main_vocab), Some(candidate_vocab)) = (main.vocab_size, candidate.vocab_size) else {
        return Err("vocabulary size is missing from one of the files".into());
    };
    if main_vocab != candidate_vocab {
        return Err(format!(
            "vocabulary size differs ({main_vocab} vs {candidate_vocab}); llama-server aborts on a draft vocab mismatch"
        ));
    }
    if let (Some(main_params), Some(candidate_params)) =
        (main.parameter_estimate, candidate.parameter_estimate)
    {
        if candidate_params >= main_params {
            return Err("the draft model is not smaller than the main model".into());
        }
    }
    let mut reasons = vec![
        format!("same architecture ({main_arch})"),
        format!("same vocabulary ({main_vocab} tokens)"),
    ];
    if let Some(parameters) = candidate.parameter_estimate {
        reasons.push(format!("{} parameters", format_parameters(parameters)));
    }
    Ok(reasons)
}

fn format_parameters(parameters: u64) -> String {
    if parameters >= 1_000_000_000 {
        format!("{:.1}B", parameters as f64 / 1_000_000_000.0)
    } else if parameters >= 1_000_000 {
        format!("{:.0}M", parameters as f64 / 1_000_000.0)
    } else {
        parameters.to_string()
    }
}

fn estimate_runtime_bytes(bytes: u64) -> u64 {
    let weights = bytes as f64;
    (weights * 1.15 + 1.5 * GIB).round() as u64
}

fn pair_with_infos(
    main: &GgufCompanionInfo,
    infos: &[GgufCompanionInfo],
    bound_mmproj: &str,
    bound_draft: &str,
    repo: Option<String>,
    active_bundle_id: Option<String>,
) -> CompanionPairing {
    let mut vision = VisionPairing::default();
    let mut draft = DraftPairing::default();
    let mut vision_bound_error: Option<String> = None;
    let mut draft_bound_error: Option<String> = None;

    if main.role == ModelRole::VisionMain {
        if !bound_mmproj.trim().is_empty() {
            if let Some(bound) = infos.iter().find(|info| {
                same_path(&info.path, bound_mmproj) && info.role == ModelRole::Mmproj
            }) {
                let family_mismatch = matches!(
                    (&main.vision_family, &bound.vision_family),
                    (Some(left), Some(right)) if left != right
                );
                if family_mismatch {
                    vision_bound_error = Some(format!(
                        "The bound projector {} is for {}, not {}. Pair a matching mmproj.",
                        bound.file_name,
                        bound.vision_family.as_deref().unwrap_or("another family"),
                        main.vision_family.as_deref().unwrap_or("this model")
                    ));
                } else {
                    vision.bound_path = Some(bound.path.clone());
                }
            }
        }

        let mut vision_candidates: Vec<(i32, bool, Vec<String>, &GgufCompanionInfo)> = infos
            .iter()
            .filter(|info| info.role == ModelRole::Mmproj && info.path != main.path)
            .map(|info| {
                let (score, compatible, reasons) = score_mmproj(main, info);
                (score, compatible, reasons, info)
            })
            .collect();
        vision_candidates.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then(b.0.cmp(&a.0))
                .then(a.3.byte_size.cmp(&b.3.byte_size))
                .then(a.3.file_name.cmp(&b.3.file_name))
        });
        vision.candidates = vision_candidates
            .into_iter()
            .map(|(score, compatible, reasons, info)| {
                candidate_from_info(info, score, compatible, reasons)
            })
            .collect();
        vision.recommended = vision
            .candidates
            .iter()
            .find(|candidate| candidate.compatible)
            .cloned();
        vision.missing = vision.bound_path.is_none();
        vision.message = if let Some(error) = vision_bound_error.as_ref() {
            error.clone()
        } else if let Some(bound) = vision.bound_path.as_ref() {
            format!(
                "Vision ready: {} is bound as the projector.",
                Path::new(bound)
                    .file_name()
                    .map(|name| name.to_string_lossy().to_string())
                    .unwrap_or_else(|| "a projector".into())
            )
        } else if let Some(recommended) = vision.recommended.as_ref() {
            format!(
                "Missing projector: {} looks compatible. Pair it before attaching images.",
                recommended.file_name
            )
        } else if main.vision_family.is_some() {
            format!(
                "Missing projector for {}: no matching mmproj GGUF is in the library.",
                main.vision_family.as_deref().unwrap_or("this vision model")
            )
        } else {
            "Missing projector: no mmproj GGUF is in the library.".into()
        };
    } else {
        vision.message = "This is a text-only model; no projector is needed.".into();
    }

    if main.role.is_main() {
        if !bound_draft.trim().is_empty() {
            if let Some(bound) = infos.iter().find(|info| {
                same_path(&info.path, bound_draft) && info.role == ModelRole::DraftCandidate
            }) {
                match draft_compatibility(main, bound) {
                    Ok(_) => {
                        draft.bound_path = Some(bound.path.clone());
                        draft.paired = true;
                    }
                    Err(reason) => {
                        draft_bound_error = Some(format!(
                            "The bound draft {} cannot be used: {reason}.",
                            bound.file_name
                        ));
                    }
                }
            }
        }

        let mut draft_candidates: Vec<(i32, bool, Vec<String>, &GgufCompanionInfo)> = infos
            .iter()
            .filter(|info| info.role == ModelRole::DraftCandidate && info.path != main.path)
            .map(|info| {
                match draft_compatibility(main, info) {
                    Ok(reasons) => {
                        let family_bonus = token_overlap(
                            &format!(
                                "{} {}",
                                main.file_name,
                                main.name.as_deref().unwrap_or_default()
                            ),
                            &format!(
                                "{} {}",
                                info.file_name,
                                info.name.as_deref().unwrap_or_default()
                            ),
                        )
                        .min(4) as i32
                            * 10;
                        let size_bonus = info
                            .parameter_estimate
                            .map(|parameters| {
                                ((parameters.min(DRAFT_PARAMETER_LIMIT) as f64
                                    / DRAFT_PARAMETER_LIMIT as f64)
                                    * 20.0) as i32
                            })
                            .unwrap_or(0);
                        let mut reasons = reasons;
                        if family_bonus > 0 {
                            reasons.push("shared family naming".into());
                        }
                        if same_directory(main, info) {
                            reasons.push("same folder".into());
                        }
                        (
                            family_bonus
                                + size_bonus
                                + if same_directory(main, info) { 10 } else { 0 },
                            true,
                            reasons,
                            info,
                        )
                    }
                    Err(reason) => (0, false, vec![reason], info),
                }
            })
            .collect();
        draft_candidates.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then(b.0.cmp(&a.0))
                .then(a.3.byte_size.cmp(&b.3.byte_size))
                .then(a.3.file_name.cmp(&b.3.file_name))
        });
        draft.candidates = draft_candidates
            .into_iter()
            .map(|(score, compatible, reasons, info)| {
                candidate_from_info(info, score, compatible, reasons)
            })
            .collect();
        draft.recommended = draft
            .candidates
            .iter()
            .find(|candidate| candidate.compatible)
            .cloned();
        let best_draft_bytes = draft
            .recommended
            .as_ref()
            .map(|candidate| candidate.size_bytes)
            .unwrap_or(0);
        draft.combined_bytes = main.byte_size.saturating_add(best_draft_bytes);
        draft.message = if let Some(error) = draft_bound_error.as_ref() {
            error.clone()
        } else if let Some(bound) = draft.bound_path.as_ref() {
            format!(
                "Speculative decoding ready: {} drafts for this model.",
                Path::new(bound)
                    .file_name()
                    .map(|name| name.to_string_lossy().to_string())
                    .unwrap_or_else(|| "a draft model".into())
            )
        } else if let Some(recommended) = draft.recommended.as_ref() {
            format!(
                "A compatible draft model is available: {}.",
                recommended.file_name
            )
        } else if main.vocab_size.is_none() {
            "No draft pairing: this file does not declare a vocabulary size, so a safe match cannot be proven.".into()
        } else if main.architecture.is_none() {
            "No draft pairing: this file does not declare an architecture, so a safe match cannot be proven.".into()
        } else {
            "No compatible draft model in the library. A draft must share the exact architecture and vocabulary size.".into()
        };
    } else {
        draft.message = "Projectors do not take a draft model.".into();
    }

    let recommended_mmproj_bytes = vision
        .recommended
        .as_ref()
        .map(|candidate| candidate.size_bytes)
        .unwrap_or(0);
    let combined_size_bytes = main
        .byte_size
        .saturating_add(recommended_mmproj_bytes)
        .saturating_add(draft.recommended.as_ref().map(|c| c.size_bytes).unwrap_or(0));

    let estimated_ram_bytes = estimate_runtime_bytes(combined_size_bytes);
    CompanionPairing {
        main: main.clone(),
        vision,
        draft,
        combined_size_bytes,
        estimated_runtime_bytes: estimated_ram_bytes,
        estimated_ram_bytes,
        estimated_vram_bytes: (combined_size_bytes as f64 * 1.05).round() as u64,
        repo,
        active_bundle_id,
    }
}

fn repo_for_path(root: &Path, path: &str) -> Option<String> {
    let manifest = crate::system::read_download_manifest(root);
    manifest.iter().find_map(|record| {
        record
            .files
            .iter()
            .any(|file| same_path(
                path,
                &root
                    .join(file.replace('/', std::path::MAIN_SEPARATOR_STR))
                    .to_string_lossy(),
            ))
            .then(|| record.repo.clone())
    })
}

fn collect_infos(paths: &[PathBuf]) -> (Vec<GgufCompanionInfo>, Vec<String>) {
    let mut infos = Vec::new();
    let mut warnings = Vec::new();
    for path in paths {
        match inspect_path(path, true) {
            Ok(info) => infos.push(info),
            Err(err) => warnings.push(err.to_string()),
        }
    }
    (infos, warnings)
}

/// Launch-time guard: report a definite architecture or vocabulary mismatch
/// between a main model and a draft. Files that cannot be inspected yield no
/// verdict, so a synthetic or partially written test file is never blocked.
pub fn draft_pair_problem(main_path: &str, draft_path: &str) -> Option<String> {
    let main = inspect_gguf_metadata(main_path).ok()?;
    let draft = inspect_gguf_metadata(draft_path).ok()?;
    let (Some(main_arch), Some(draft_arch)) =
        (main.architecture.as_deref(), draft.architecture.as_deref())
    else {
        return None;
    };
    if !main_arch.eq_ignore_ascii_case(draft_arch) {
        return Some(format!(
            "architecture differs ({main_arch} vs {draft_arch})"
        ));
    }
    if let (Some(main_vocab), Some(draft_vocab)) = (main.vocab_size, draft.vocab_size) {
        if main_vocab != draft_vocab {
            return Some(format!(
                "vocabulary size differs ({main_vocab} vs {draft_vocab}); llama-server aborts on a draft vocab mismatch"
            ));
        }
    }
    None
}

/// Launch-time guard for a manually configured projector. Only a definite,
/// known cross-family pairing is rejected; unknown families are left to
/// llama.cpp, which reports its own error.
pub fn vision_pair_problem(main_path: &str, mmproj_path: &str) -> Option<String> {
    let main = inspect_gguf_metadata(main_path).ok()?;
    let projector = inspect_gguf_metadata(mmproj_path).ok()?;
    if projector.role != ModelRole::Mmproj {
        return Some(format!(
            "{} is a {}, not a vision projector",
            projector.file_name,
            role_label(projector.role)
        ));
    }
    match (main.vision_family.as_deref(), projector.vision_family.as_deref()) {
        (Some(main_family), Some(projector_family)) if main_family != projector_family => {
            Some(format!(
                "the projector is for {projector_family}, but the model is {main_family}"
            ))
        }
        _ => None,
    }
}

pub fn find_compatible_companions(
    main_path: &str,
    library: &[String],
    bound_mmproj: &str,
    bound_draft: &str,
) -> AppResult<CompanionPairing> {
    let main_expanded = expand_home(main_path.trim());
    let main = inspect_path(&main_expanded, true)?;
    let mut infos = Vec::new();
    for path in library {
        let expanded = expand_home(path);
        if expanded == main_expanded {
            continue;
        }
        if let Ok(info) = inspect_path(&expanded, true) {
            infos.push(info);
        }
    }
    infos.push(main.clone());

    Ok(pair_with_infos(
        &main,
        &infos,
        bound_mmproj,
        bound_draft,
        repo_for_path(
            main_expanded
                .parent()
                .unwrap_or_else(|| Path::new(".")),
            &main.path,
        ),
        None,
    ))
}

fn normalize_bundle(mut bundle: ModelBundle) -> ModelBundle {
    bundle.main_model_path = bundle.main_model_path.trim().to_string();
    bundle.mmproj_path = bundle.mmproj_path.trim().to_string();
    bundle.draft_model_path = bundle.draft_model_path.trim().to_string();
    bundle.draft_max_tokens = bundle.draft_max_tokens.clamp(1, 64);
    bundle.draft_min_tokens = bundle.draft_min_tokens.min(bundle.draft_max_tokens);
    if !bundle.draft_p_min.is_finite() {
        bundle.draft_p_min = 0.75;
    }
    bundle.draft_p_min = bundle.draft_p_min.clamp(0.0, 1.0);
    if bundle.name.trim().is_empty() {
        bundle.name = Path::new(&bundle.main_model_path)
            .file_stem()
            .map(|stem| stem.to_string_lossy().to_string())
            .filter(|stem| !stem.is_empty())
            .unwrap_or_else(|| "Model bundle".into());
    }
    bundle
}

// ---------------------------------------------------------------------------
// Bundle validation
// ---------------------------------------------------------------------------

pub fn validate_bundle(bundle: &ModelBundle) -> BundleCheck {
    let bundle = normalize_bundle(bundle.clone());
    let mut check = BundleCheck::default();
    if bundle.main_model_path.is_empty() {
        check.errors.push("Choose a main model for the bundle.".into());
        return check;
    }

    let main = match inspect_gguf_metadata(&bundle.main_model_path) {
        Ok(info) => info,
        Err(err) => {
            check.errors.push(err.to_string());
            return check;
        }
    };
    if main.role == ModelRole::Mmproj {
        check.errors.push(format!(
            "{} is a vision projector, not a chat model.",
            main.file_name
        ));
    }
    if main.role == ModelRole::DraftCandidate {
        check.warnings.push(format!(
            "{} is a small model; pairing a draft for it is unusual.",
            main.file_name
        ));
    }
    check.combined_size_bytes = main.byte_size;
    check.main = Some(main.clone());

    if !bundle.mmproj_path.is_empty() {
        if main.role != ModelRole::VisionMain {
            check.errors.push(format!(
                "{} is a {}, so it cannot use a vision projector.",
                main.file_name,
                role_label(main.role)
            ));
        }
        match inspect_gguf_metadata(&bundle.mmproj_path) {
            Ok(mmproj) => {
                if mmproj.role != ModelRole::Mmproj {
                    check.errors.push(format!(
                        "{} is a {}, not a vision projector.",
                        mmproj.file_name,
                        role_label(mmproj.role)
                    ));
                }
                if let (Some(main_family), Some(mmproj_family)) =
                    (main.vision_family.as_deref(), mmproj.vision_family.as_deref())
                {
                    if main_family != mmproj_family {
                        check.errors.push(format!(
                            "The projector is for {mmproj_family}, but the main model is {main_family}. llama-server cannot mix projector families."
                        ));
                    }
                }
                check.combined_size_bytes = check.combined_size_bytes.saturating_add(mmproj.byte_size);
                check.mmproj = Some(mmproj);
            }
            Err(err) => check.errors.push(err.to_string()),
        }
    } else if main.role == ModelRole::VisionMain {
        check.warnings.push(
            "This vision model has no projector bound. Images will be rejected until an mmproj is paired.".into(),
        );
    }

    if !bundle.draft_model_path.is_empty() {
        match inspect_gguf_metadata(&bundle.draft_model_path) {
            Ok(draft) => {
                match draft_compatibility(&main, &draft) {
                    Ok(_) => {}
                    Err(reason) => check.errors.push(format!(
                        "The draft model cannot be paired: {reason}."
                    )),
                }
                check.combined_size_bytes = check.combined_size_bytes.saturating_add(draft.byte_size);
                check.draft = Some(draft);
            }
            Err(err) => check.errors.push(err.to_string()),
        }
    }

    if bundle.draft_max_tokens == 0 {
        check.warnings.push("Draft max tokens is 0; llama.cpp will use its own default.".into());
    }
    check.estimated_runtime_bytes = estimate_runtime_bytes(check.combined_size_bytes);
    check.ok = check.errors.is_empty();
    check
}

// ---------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn inspect_gguf_companion(path: String) -> AppResult<GgufCompanionInfo> {
    inspect_gguf_metadata(&path)
}

#[tauri::command]
pub fn inspect_model_companions(
    store: State<'_, std::sync::Arc<SettingsStore>>,
    path: String,
) -> AppResult<CompanionPairing> {
    let settings = store.get();
    let root = expand_home(&settings.models_dir);
    let expanded = expand_home(path.trim());
    if !expanded.is_file() {
        return Err(AppError::message(format!(
            "{} does not exist. Choose a local GGUF model first.",
            expanded.display()
        )));
    }
    let mut paths = scan_gguf_files(&root);
    if !paths.contains(&expanded) {
        paths.push(expanded.clone());
    }
    let library: Vec<String> = paths
        .iter()
        .map(|path| path.to_string_lossy().to_string())
        .collect();
    let mut pairing = find_compatible_companions(
        &expanded.to_string_lossy(),
        &library,
        &settings.llama_mmproj_path,
        &settings.llama_draft_path,
    )?;
    pairing.repo = repo_for_path(&root, &pairing.main.path);
    pairing.active_bundle_id = settings
        .model_bundles
        .iter()
        .find(|bundle| same_path(&bundle.main_model_path, &pairing.main.path))
        .map(|bundle| bundle.id.clone());
    Ok(pairing)
}

#[tauri::command]
pub fn library_companions(
    store: State<'_, std::sync::Arc<SettingsStore>>,
) -> LibraryCompanions {
    let settings = store.get();
    let root = expand_home(&settings.models_dir);
    let paths = scan_gguf_files(&root);
    let (infos, warnings) = collect_infos(&paths);
    let pairings = infos
        .iter()
        .filter(|info| info.role.is_main())
        .map(|main| {
            let active_bundle = settings
                .model_bundles
                .iter()
                .find(|bundle| same_path(&bundle.main_model_path, &main.path));
            pair_with_infos(
                main,
                &infos,
                &settings.llama_mmproj_path,
                &settings.llama_draft_path,
                repo_for_path(&root, &main.path),
                active_bundle.map(|bundle| bundle.id.clone()),
            )
        })
        .collect();
    LibraryCompanions {
        models: infos,
        pairings,
        warnings,
    }
}

#[tauri::command]
pub fn validate_model_bundle(bundle: ModelBundle) -> BundleCheck {
    validate_bundle(&bundle)
}

fn apply_bundle(settings: &mut AppSettings, bundle: &ModelBundle) -> AppResult<()> {
    if bundle.main_model_path.trim().is_empty() {
        return Err(AppError::message("The bundle has no main model."));
    }
    let main = expand_home(&bundle.main_model_path);
    if !main.is_file() {
        return Err(AppError::message(format!(
            "The bundle's main model does not exist: {}",
            main.display()
        )));
    }
    settings.llama_model_path = main.to_string_lossy().to_string();
    settings.llama_mmproj_path = if bundle.mmproj_path.trim().is_empty() {
        String::new()
    } else {
        let path = expand_home(&bundle.mmproj_path);
        if !path.is_file() {
            return Err(AppError::message(format!(
                "The bundle's projector does not exist: {}",
                path.display()
            )));
        }
        path.to_string_lossy().to_string()
    };
    settings.llama_draft_path = if bundle.draft_model_path.trim().is_empty() {
        String::new()
    } else {
        let path = expand_home(&bundle.draft_model_path);
        if !path.is_file() {
            return Err(AppError::message(format!(
                "The bundle's draft model does not exist: {}",
                path.display()
            )));
        }
        path.to_string_lossy().to_string()
    };
    // A plain draft model is not an MTP head. Leaving the mode on `mtp` would
    // launch it with `--spec-type draft-mtp` and produce nonsense.
    if !settings.llama_draft_path.is_empty() {
        settings.llama_draft_path = settings.llama_draft_path.trim().to_string();
        settings.llama_mtp_mode = "auto".into();
        settings.llama_mtp_path = String::new();
    } else {
        // An MTP head file belongs to the previous main model; never inherit it.
        settings.llama_mtp_path.clear();
    }
    settings.llama_draft_max = bundle.draft_max_tokens.clamp(1, 64);
    settings.llama_draft_min = bundle.draft_min_tokens.min(settings.llama_draft_max);
    settings.llama_draft_p_min = if bundle.draft_p_min.is_finite() {
        bundle.draft_p_min.clamp(0.0, 1.0)
    } else {
        0.75
    };
    Ok(())
}

#[tauri::command]
pub fn save_model_bundle(
    store: State<'_, std::sync::Arc<SettingsStore>>,
    bundle: ModelBundle,
) -> AppResult<AppSettings> {
    let mut bundle = normalize_bundle(bundle);
    let check = validate_bundle(&bundle);
    if !check.ok {
        return Err(AppError::message(check.errors.join(" ")));
    }
    let mut settings = store.get();
    if bundle.id.trim().is_empty() {
        bundle.id = uuid::Uuid::new_v4().to_string();
        bundle.created_at_ms = now_ms();
    }
    bundle.updated_at_ms = now_ms();
    if let Some(existing) = settings
        .model_bundles
        .iter_mut()
        .find(|existing| existing.id == bundle.id)
    {
        if bundle.created_at_ms == 0 {
            bundle.created_at_ms = existing.created_at_ms;
        }
        *existing = bundle.clone();
    } else {
        settings.model_bundles.push(bundle.clone());
    }
    settings.active_bundle_id = bundle.id.clone();
    apply_bundle(&mut settings, &bundle)?;
    store.save(settings)
}

#[tauri::command]
pub fn delete_model_bundle(
    store: State<'_, std::sync::Arc<SettingsStore>>,
    id: String,
) -> AppResult<AppSettings> {
    let mut settings = store.get();
    settings.model_bundles.retain(|bundle| bundle.id != id);
    if settings.active_bundle_id == id {
        settings.active_bundle_id.clear();
    }
    store.save(settings)
}

#[tauri::command]
pub fn apply_model_bundle(
    store: State<'_, std::sync::Arc<SettingsStore>>,
    id: String,
) -> AppResult<AppSettings> {
    let mut settings = store.get();
    let bundle = settings
        .model_bundles
        .iter()
        .find(|bundle| bundle.id == id)
        .cloned()
        .ok_or_else(|| AppError::message("That model bundle no longer exists."))?;
    apply_bundle(&mut settings, &bundle)?;
    settings.active_bundle_id = bundle.id;
    store.save(settings)
}

/// One-click pairing used by the composer guardrail and the library drawer.
#[tauri::command(rename_all = "snake_case")]
pub fn pair_model_companions(
    store: State<'_, std::sync::Arc<SettingsStore>>,
    main_path: String,
    mmproj_path: Option<String>,
    draft_path: Option<String>,
    draft_max_tokens: Option<u32>,
) -> AppResult<AppSettings> {
    let mut settings = store.get();
    let main = expand_home(main_path.trim());
    if !main.is_file() {
        return Err(AppError::message(format!(
            "The main model does not exist: {}",
            main.display()
        )));
    }
    let main_text = main.to_string_lossy().to_string();
    let existing = settings
        .model_bundles
        .iter()
        .find(|bundle| same_path(&bundle.main_model_path, &main_text))
        .cloned();
    let bundle = ModelBundle {
        id: existing
            .as_ref()
            .map(|bundle| bundle.id.clone())
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        name: existing
            .as_ref()
            .map(|bundle| bundle.name.clone())
            .filter(|name| !name.trim().is_empty())
            .unwrap_or_default(),
        main_model_path: main.to_string_lossy().to_string(),
        mmproj_path: mmproj_path.unwrap_or_default(),
        draft_model_path: draft_path.unwrap_or_default(),
        draft_max_tokens: draft_max_tokens.unwrap_or(settings.llama_draft_max).clamp(1, 64),
        draft_min_tokens: settings.llama_draft_min,
        draft_p_min: settings.llama_draft_p_min,
        created_at_ms: existing
            .as_ref()
            .map(|bundle| bundle.created_at_ms)
            .unwrap_or_else(now_ms),
        updated_at_ms: now_ms(),
    };
    let bundle = normalize_bundle(bundle);
    let check = validate_bundle(&bundle);
    if !check.ok {
        return Err(AppError::message(check.errors.join(" ")));
    }
    if let Some(existing) = settings
        .model_bundles
        .iter_mut()
        .find(|existing| existing.id == bundle.id)
    {
        *existing = bundle.clone();
    } else {
        settings.model_bundles.push(bundle.clone());
    }
    settings.active_bundle_id = bundle.id.clone();
    apply_bundle(&mut settings, &bundle)?;
    store.save(settings)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(tensor_count.to_le_bytes());
        bytes.extend(metadata_count.to_le_bytes());
        bytes
    }

    fn push_key(bytes: &mut Vec<u8>, key: &str) {
        bytes.extend((key.len() as u64).to_le_bytes());
        bytes.extend(key.as_bytes());
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend((value.len() as u64).to_le_bytes());
        bytes.extend(value.as_bytes());
    }

    fn push_kv_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
        push_key(bytes, key);
        bytes.extend(TYPE_STRING.to_le_bytes());
        push_string(bytes, value);
    }

    fn push_kv_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
        push_key(bytes, key);
        bytes.extend(TYPE_UINT32.to_le_bytes());
        bytes.extend(value.to_le_bytes());
    }

    fn push_tokens(bytes: &mut Vec<u8>, key: &str, tokens: &[&str]) {
        push_key(bytes, key);
        bytes.extend(TYPE_ARRAY.to_le_bytes());
        bytes.extend(TYPE_STRING.to_le_bytes());
        bytes.extend((tokens.len() as u64).to_le_bytes());
        for token in tokens {
            push_string(bytes, token);
        }
    }

    fn push_tensor(bytes: &mut Vec<u8>, name: &str, dims: &[u64], ggml_type: u32) {
        push_key(bytes, name);
        bytes.extend((dims.len() as u32).to_le_bytes());
        for dim in dims {
            bytes.extend(dim.to_le_bytes());
        }
        bytes.extend(ggml_type.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
    }

    fn parse(bytes: Vec<u8>) -> ParsedHeader {
        read_gguf(&mut Cursor::new(bytes)).expect("valid synthetic GGUF")
    }

    fn info(
        path: &str,
        role: ModelRole,
        architecture: Option<&str>,
        vocab: Option<u32>,
        params: u64,
        size: u64,
        family: Option<&str>,
    ) -> GgufCompanionInfo {
        GgufCompanionInfo {
            path: path.into(),
            file_name: Path::new(path)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string(),
            role,
            architecture: architecture.map(str::to_string),
            name: None,
            tokenizer_model: Some("gpt2".into()),
            vocab_size: vocab,
            context_length: Some(32_768),
            parameter_estimate: Some(params),
            byte_size: size,
            quant: "Q4_K_M".into(),
            tensor_count: 10,
            vision_tensor_count: 0,
            vision_markers: Vec::new(),
            vision_family: family.map(str::to_string),
        }
    }

    #[test]
    fn reads_vocab_context_and_parameters_without_touching_weights() {
        let mut bytes = header(3, 6);
        push_kv_string(&mut bytes, "general.architecture", "qwen2");
        push_kv_string(&mut bytes, "general.name", "Qwen2.5 7B Instruct");
        push_kv_string(&mut bytes, "tokenizer.ggml.model", "gpt2");
        push_kv_u32(&mut bytes, "qwen2.context_length", 32_768);
        push_tokens(&mut bytes, "tokenizer.ggml.tokens", &["a", "b", "c", "d"]);
        push_kv_u32(&mut bytes, "qwen2.block_count", 28);
        push_tensor(&mut bytes, "token_embd.weight", &[8, 4], 12);
        push_tensor(&mut bytes, "blk.0.attn_q.weight", &[8, 8], 12);
        push_tensor(&mut bytes, "output.weight", &[8, 4], 12);
        // Tensor payloads are intentionally absent: the parser must not read them.

        let parsed = parse(bytes);
        assert_eq!(parsed.architecture.as_deref(), Some("qwen2"));
        assert_eq!(parsed.vocab_size, Some(4));
        assert_eq!(parsed.context_length, Some(32_768));
        assert_eq!(parsed.tensor_count, 3);
        assert_eq!(parsed.parameter_estimate, 32 + 64 + 32);
        assert!(!parsed.clip_architecture);
    }

    #[test]
    fn skips_large_metadata_arrays_and_finds_later_keys() {
        let mut bytes = header(1, 3);
        push_tokens(
            &mut bytes,
            "tokenizer.ggml.tokens",
            &["alpha", "beta", "gamma", "delta", "epsilon"],
        );
        push_kv_string(&mut bytes, "llama.rope.scaling.type", "linear");
        push_kv_u32(&mut bytes, "llama.context_length", 8_192);
        push_tensor(&mut bytes, "token_embd.weight", &[16, 5], 1);

        let parsed = parse(bytes);
        assert_eq!(parsed.vocab_size, Some(5));
        assert_eq!(parsed.context_length, Some(8_192));
    }

    #[test]
    fn classifies_a_clip_projector_from_metadata_and_tensors() {
        let mut bytes = header(3, 4);
        push_kv_string(&mut bytes, "general.architecture", "clip");
        push_kv_string(&mut bytes, "general.name", "Qwen2.5-VL mmproj");
        push_kv_string(&mut bytes, "clip.projector_type", "qwen2vl_merger");
        push_kv_u32(&mut bytes, "clip.vision.image_size", 448);
        push_tensor(&mut bytes, "mm.0.weight", &[8, 8], 1);
        push_tensor(&mut bytes, "v.blk.0.attn_q.weight", &[8, 8], 1);
        push_tensor(&mut bytes, "v.position_embd.weight", &[8, 8], 1);

        let parsed = parse(bytes);
        assert!(parsed.clip_architecture);
        assert_eq!(parsed.vocab_size, None);
        assert_eq!(parsed.vision_tensor_count, 3);
        let (role, family) = classify(
            "mmproj-model-f16.gguf",
            parsed.architecture.as_deref(),
            parsed.name.as_deref(),
            parsed.vocab_size,
            parsed.vision_tensor_count,
            &parsed.vision_markers,
            parsed.clip_architecture,
            Some(parsed.parameter_estimate),
        );
        assert_eq!(role, ModelRole::Mmproj);
        assert_eq!(family.as_deref(), Some("qwen2.5-vl"));
    }

    #[test]
    fn classifies_vision_main_and_plain_main() {
        let vision = classify(
            "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
            Some("qwen2vl"),
            Some("Qwen2.5-VL-7B-Instruct"),
            Some(152_064),
            0,
            &[],
            false,
            Some(7_600_000_000),
        );
        assert_eq!(vision.0, ModelRole::VisionMain);
        assert_eq!(vision.1.as_deref(), Some("qwen2.5-vl"));

        let plain = classify(
            "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            Some("qwen2"),
            Some("Qwen2.5 7B Instruct"),
            Some(152_064),
            0,
            &[],
            false,
            Some(7_600_000_000),
        );
        assert_eq!(plain.0, ModelRole::MainLlm);
        assert!(plain.1.is_none());
    }

    #[test]
    fn gemma_3_1b_is_text_but_4b_needs_a_projector() {
        let small = classify(
            "gemma-3-1b-it-Q4_K_M.gguf",
            Some("gemma3"),
            Some("gemma-3-1b-it"),
            Some(262_144),
            0,
            &[],
            false,
            Some(1_000_000_000),
        );
        assert_eq!(small.0, ModelRole::DraftCandidate);

        let large = classify(
            "gemma-3-4b-it-Q4_K_M.gguf",
            Some("gemma3"),
            Some("gemma-3-4b-it"),
            Some(262_144),
            0,
            &[],
            false,
            Some(4_300_000_000),
        );
        assert_eq!(large.0, ModelRole::VisionMain);
        assert_eq!(large.1.as_deref(), Some("gemma-3"));
    }

    #[test]
    fn rejects_non_gguf_and_huge_headers() {
        assert!(read_gguf(&mut Cursor::new(b"not a gguf".to_vec())).is_err());
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend((MAX_METADATA_ENTRIES + 1).to_le_bytes());
        let error = read_gguf(&mut Cursor::new(bytes)).unwrap_err();
        assert!(error.contains("unreasonable"), "{error}");
    }

    #[test]
    fn draft_pairing_requires_exact_architecture_and_vocabulary() {
        let main = info(
            "/models/qwen/main.gguf",
            ModelRole::MainLlm,
            Some("qwen2"),
            Some(151_936),
            7_000_000_000,
            4_000,
            None,
        );
        let good = info(
            "/models/qwen/draft.gguf",
            ModelRole::DraftCandidate,
            Some("qwen2"),
            Some(151_936),
            500_000_000,
            300,
            None,
        );
        let wrong_vocab = info(
            "/models/qwen/draft2.gguf",
            ModelRole::DraftCandidate,
            Some("qwen2"),
            Some(151_936 - 1),
            500_000_000,
            300,
            None,
        );
        let wrong_arch = info(
            "/models/qwen/draft3.gguf",
            ModelRole::DraftCandidate,
            Some("llama"),
            Some(151_936),
            500_000_000,
            300,
            None,
        );

        assert!(draft_compatibility(&main, &good).is_ok());
        let vocab_error = draft_compatibility(&main, &wrong_vocab).unwrap_err();
        assert!(vocab_error.contains("vocabulary size differs"), "{vocab_error}");
        let arch_error = draft_compatibility(&main, &wrong_arch).unwrap_err();
        assert!(arch_error.contains("architecture differs"), "{arch_error}");
    }

    #[test]
    fn vision_pairing_prefers_same_family_and_flags_missing() {
        let main = info(
            "/models/qwen/main.gguf",
            ModelRole::VisionMain,
            Some("qwen2vl"),
            Some(151_936),
            7_000_000_000,
            4_000,
            Some("qwen2.5-vl"),
        );
        let matching = info(
            "/models/qwen/mmproj-model-f16.gguf",
            ModelRole::Mmproj,
            Some("clip"),
            None,
            600_000_000,
            900,
            Some("qwen2.5-vl"),
        );
        let mismatched = info(
            "/models/other/mmproj-llava-f16.gguf",
            ModelRole::Mmproj,
            Some("clip"),
            None,
            600_000_000,
            900,
            Some("llava"),
        );
        let infos = vec![main.clone(), matching.clone(), mismatched.clone()];

        let pairing = pair_with_infos(&main, &infos, "", "", None, None);
        assert!(pairing.vision.missing);
        assert_eq!(
            pairing
                .vision
                .recommended
                .as_ref()
                .map(|candidate| candidate.path.as_str()),
            Some(matching.path.as_str())
        );
        assert!(pairing.vision.candidates[0].compatible);
        assert!(!pairing
            .vision
            .candidates
            .iter()
            .find(|candidate| candidate.path == mismatched.path)
            .unwrap()
            .compatible);

        let bound = pair_with_infos(&main, &infos, &matching.path, "", None, None);
        assert!(!bound.vision.missing);
        assert_eq!(bound.vision.bound_path.as_deref(), Some(matching.path.as_str()));
    }

    #[test]
    fn bundle_validation_blocks_a_vocab_mismatch() {
        let root = std::env::temp_dir().join(format!(
            "janas-bundle-test-{}-{}",
            std::process::id(),
            now_ms()
        ));
        fs::create_dir_all(&root).expect("create test root");

        let main_path = root.join("main.gguf");
        let draft_path = root.join("draft.gguf");
        let mmproj_path = root.join("mmproj-model-f16.gguf");

        let mut main = header(1, 6);
        push_kv_string(&mut main, "general.architecture", "qwen2");
        push_kv_string(&mut main, "general.name", "Qwen2.5-VL-7B-Instruct");
        push_kv_string(&mut main, "tokenizer.ggml.model", "gpt2");
        push_kv_u32(&mut main, "qwen2.context_length", 32_768);
        push_tokens(&mut main, "tokenizer.ggml.tokens", &["a", "b"]);
        push_kv_string(&mut main, "qwen2.vision.projector", "qwen2vl");
        push_tensor(&mut main, "token_embd.weight", &[8, 2], 12);
        fs::write(&main_path, main).unwrap();

        let mut draft = header(1, 2);
        push_kv_string(&mut draft, "general.architecture", "qwen2");
        push_tokens(&mut draft, "tokenizer.ggml.tokens", &["a", "b", "c"]);
        push_tensor(&mut draft, "token_embd.weight", &[8, 3], 12);
        fs::write(&draft_path, draft).unwrap();

        let mut mmproj = header(1, 2);
        push_kv_string(&mut mmproj, "general.architecture", "clip");
        push_kv_string(&mut mmproj, "general.name", "Qwen2.5-VL mmproj");
        push_tensor(&mut mmproj, "mm.0.weight", &[4, 4], 1);
        fs::write(&mmproj_path, mmproj).unwrap();

        let bundle = ModelBundle {
            main_model_path: main_path.to_string_lossy().to_string(),
            mmproj_path: mmproj_path.to_string_lossy().to_string(),
            draft_model_path: draft_path.to_string_lossy().to_string(),
            draft_max_tokens: 8,
            ..ModelBundle::default()
        };
        let check = validate_bundle(&bundle);
        assert!(!check.ok);
        assert!(
            check
                .errors
                .iter()
                .any(|error| error.contains("vocabulary size differs")),
            "{:?}",
            check.errors
        );

        let clean = ModelBundle {
            draft_model_path: String::new(),
            mmproj_path: String::new(),
            ..bundle
        };
        let check = validate_bundle(&clean);
        assert!(check.ok, "{:?}", check.errors);
        assert!(check.warnings.iter().any(|warning| warning.contains("projector")));
        assert!(check.combined_size_bytes > 0);

        let _ = fs::remove_dir_all(&root);
    }
}
