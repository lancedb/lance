
const FSST_MAGIC: u64 = 0x46535354 << 32; // "FSST"
const FSST_ESC: u8 = 255;
const FSST_CODE_BITS: u16 = 9;
// first 256 codes [0,255] are pseudo codes: escaped bytes
const FSST_CODE_BASE: u16 = 256;
const FSST_CODE_MAX: u16 = 1 << FSST_CODE_BITS;
const FSST_CODE_UNSET: u16 = FSST_CODE_MAX;
// all code bits set
const FSST_CODE_MASK: u16 = FSST_CODE_MAX - 1;
// we construct FSST symbol tables using a random sample of about 16KB (1<<14) 
const FSST_SAMPLETARGET: usize = 1 << 14;
const FSST_SAMPLEMAXSZ: usize = 2 * FSST_SAMPLETARGET;
//const FSST_LEAST_INPUT_SIZE: usize = 8 * 1024 * 1024;   // 8MB 
// set low in development 
const FSST_LEAST_INPUT_SIZE: usize = 2;   // 8MB 

const FSST_ICL_FREE: u32 = (8 << 28) | ((FSST_CODE_MASK as u32) << 16);


const FSST_HASH_LOG2SIZE: usize = 10;
const FSST_HASH_PRIME: u64 = 2971215073;
const FSST_SHIFT: usize = 15;
const FSST_HASH: fn(u64) -> u64 = |w| ((w.wrapping_mul(FSST_HASH_PRIME)^((w.wrapping_mul(FSST_HASH_PRIME)))>>FSST_SHIFT));
const MAX_SYMBOL_LENGTH: usize = 8;

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::io;
use std::ptr;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn fsst_unaligned_load_unchecked(v: *const u8) -> u64 {
    unsafe { ptr::read_unaligned(v as *const u64) }
}

#[derive(Default, Copy, Clone, PartialEq, Eq)]
struct Symbol {
    // the byte sequence that this symbol stands for
    val: u64, // usually we process it as a num(ber), as this is fast

    // icl = u64 ignoredBits:16,code:12,length:4,unused:32 -- but we avoid exposing this bit-field notation
    // use a single u64 to be sure "code" is accessed with one load and can be compared with one comparison
    icl: u64,  
}

use std::fmt;

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = self.val.to_ne_bytes();
        for i in 0..self.length() {
            write!(f, "{}", bytes[i as usize] as char)?;
        }
        write!(f, "\t")?;
        write!(f, "ignoredBits: {}, code: {}, length: {}", self.ignored_bits(), self.code(), self.length())?;
        Ok(())
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = self.val.to_ne_bytes();
        for i in 0..self.length() {
            write!(f, "{}", bytes[i as usize] as char)?;
        }
        write!(f, "\t")?;
        write!(f, "ignoredBits: {}, code: {}, length: {}", self.ignored_bits(), self.code(), self.length())?;
        Ok(())
    }
}

impl Symbol {
    fn new() -> Self {
        Self { val: 0, icl: FSST_ICL_FREE as u64}
    }

    fn from_char(c: u8, code: u16) -> Self {
        Self { val: c as u64, icl: (1<<28)|(code as u64)<<16|56 } // 56 = 8*7, in a symbol which represents a single 
                                                                  // character, 56 bits are ignored
    }

    fn set_code_len(&mut self, code: u16, len: u32) {
        self.icl = ((len as u64)<< 28)|((code as u64) << 16)|((8u64.saturating_sub(len as u64))*8);
    }

    fn length(&self) -> u32 {
        (self.icl >> 28) as u32
    }

    fn code(&self) -> u16 {
        ((self.icl >> 16) & FSST_CODE_MASK as u64) as u16
    }

    // ignoredBits is (8-length)*8, which is the amount of high bits to zero in the input word before comparing with the hashtable key
    // it could of course be computed from len during lookup, but storing it precomputed in some loose bits is faster
    //
    fn ignored_bits(&self) -> u32 {
        (self.icl & u16::MAX as u64) as u32
    }

    fn first(&self) -> u8 {
        assert!(self.length() >= 1);
        (0xFF & self.val) as u8
    }

    fn first2(&self) -> u16 {
        assert!(self.length() >= 2);
        (0xFFFF & self.val) as u16
    }

    fn hash(&self) -> u64 {
        let v = 0xFFFFFF & self.val;
        (FSST_HASH)(v)
    }

    // right is the substring follows left
    // for example, in "hello", 
    // "llo" is the substring that follows "he"
    fn concat(left: Self, right: Self) -> Self {
        let mut s = Self::new();
        let mut length = left.length() + right.length();
        if length > MAX_SYMBOL_LENGTH as u32 {
            length = MAX_SYMBOL_LENGTH as u32;
        }
        s.set_code_len(FSST_CODE_MASK as u16, length);
        s.val = (right.val << (8 * left.length())) | left.val;
        s
    }
}

// Symbol that can be put in a queue, ordered on gain
#[derive(Clone)]
struct QSymbol {
    symbol: Symbol,
    // the gain field is only used in the symbol queue that sorts symbols on gain
    gain: u32, 
}

impl PartialEq for QSymbol {
    fn eq(&self, other: &Self) -> bool {
        self.symbol.val == other.symbol.val && self.symbol.icl == other.symbol.icl
    }
}

impl Ord for QSymbol {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.gain.cmp(&other.gain)
            .then_with(|| other.symbol.val.cmp(&self.symbol.val)) 
    }
}

impl PartialOrd for QSymbol {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for QSymbol {}

use std::hash::{Hash, Hasher};

impl Hash for QSymbol {
    // this hash algorithm follows the C++ implementation of the FSST in the paper
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut k = self.symbol.val;
        const M: u64 = 0xc6a4a7935bd1e995;
        const R: u32 = 47;
        let mut h: u64 = 0x8445d61a4e774912 ^ (8u64.wrapping_mul(M));
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);
        h ^= k;
        h = h.wrapping_mul(M);
        h ^= h >> R;
        h = h.wrapping_mul(M);
        h ^= h >> R;
        h.hash(state);
    }
}

#[derive(Clone)]
struct SymbolTable {
    short_codes: [u16; 65536],
    byte_codes: [u16; 256],
    symbols: [Symbol; FSST_CODE_MAX as usize],
    hash_tab: [Symbol; 1 << FSST_HASH_LOG2SIZE as usize],
    hash_tab_size: usize,
    n_symbols: u16,
    terminator: u16,
    // in a finalized symbol table, symbols are arranged by their symbol length, 
    // in the order of 2, 3, 4, 5, 6, 7, 8, 1, codes < suffix_lim are 2 bytes codes that don't have a longer suffix
    suffix_lim: u16, 
    len_histo: [u8; FSST_CODE_BITS as usize],
}

impl std::fmt::Display for SymbolTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A FSST SymbolTable after finalize():\n")?;
        write!(f, "n_symbols: {}\n", self.n_symbols)?;
        for i in 0 as usize..self.n_symbols as usize {
            write!(f, "symbols[{}]: {}\n", i, self.symbols[i])?;
        }
        write!(f, "suffix_lim: {}\n", self.suffix_lim)?;
        for i in 0..FSST_CODE_BITS {
            write!(f, "len_histo[{}]: {}\n", i, self.len_histo[i as usize])?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for SymbolTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A FSST SymbolTable before finalize():\n")?;
        write!(f, "n_symbols: {}\n", self.n_symbols)?;
        for i in FSST_CODE_BASE as usize..FSST_CODE_BASE as usize + self.n_symbols as usize {
            write!(f, "symbols[{}]: {}\n", i, self.symbols[i])?;
        }
        write!(f, "suffix_lim: {}\n", self.suffix_lim)?;
        for i in 0..FSST_CODE_BITS {
            write!(f, "len_histo[{}]: {}\n", i, self.len_histo[i as usize])?;
        }
        Ok(())
    }
}

impl SymbolTable {
    fn new() -> Self {
        let mut symbols = [Symbol::new(); FSST_CODE_MAX as usize];
        for i in 0..256 {
            symbols[i] = Symbol::from_char(i as u8, i as u16);
        }
        let unused = Symbol::from_char(0, FSST_CODE_MASK as u16);
        for i in 256..FSST_CODE_MAX {
            symbols[i as usize] = unused;
        }
        let s = Symbol::new();
        let hash_tab = [s; 1 << FSST_HASH_LOG2SIZE];
        let mut byte_codes = [0; 256];
        for i in 0..256 {
            byte_codes[i] = i as u16;
        }
        let mut short_codes = [FSST_CODE_MASK; 65536];
        for i in 0..=65535 as u16 {
            short_codes[i as usize] = i & 0xFF; 
        }
        Self {
            short_codes,
            byte_codes,
            symbols,
            hash_tab,
            hash_tab_size: 1 << FSST_HASH_LOG2SIZE,
            n_symbols: 0,
            terminator: 256,
            suffix_lim: FSST_CODE_MAX as u16,
            len_histo: [0; FSST_CODE_BITS as usize],
        }
    } 

    fn clear(&mut self) {
        for i in 0..256 {
            self.symbols[i] = Symbol::from_char(i as u8, i as u16);
        }
        let unused = Symbol::from_char(0, FSST_CODE_MASK as u16);
        for i in 256..FSST_CODE_MAX {
            self.symbols[i as usize] = unused;
        }
        for i in 0..256 {
            self.byte_codes[i] = i as u16;
        }
        for i in 0..=65535 as u16 {
            self.short_codes[i as usize] = i & 0xFF;
        }
        let s = Symbol::new();
        for i in 0..1 << FSST_HASH_LOG2SIZE {
            self.hash_tab[i] = s;
        }
        for i in 0..FSST_CODE_BITS as usize {
            self.len_histo[i] = 0;
        }
        self.n_symbols = 0;
    }

    fn hash_insert(&mut self, s: Symbol) -> bool {
        let idx = (s.hash() & (self.hash_tab_size as u64 - 1)) as usize;
        let taken = self.hash_tab[idx].icl < FSST_ICL_FREE as u64;
        if taken {
            return false; // collision in hash table
        }
        //println!("inserting a hash symbl: {}", s);
        self.hash_tab[idx].icl = s.icl;
        self.hash_tab[idx].val = s.val & (u64::MAX >> (s.ignored_bits()));
        true
    }

    fn add(&mut self, mut s: Symbol) -> bool {
        assert!(FSST_CODE_BASE as u16 + self.n_symbols < FSST_CODE_MAX as u16);
        let len = s.length();
        s.set_code_len(FSST_CODE_BASE as u16 + self.n_symbols, len);
        if len == 1 {
            self.byte_codes[s.first() as usize] = FSST_CODE_BASE + self.n_symbols;
        } else if len == 2 {
            self.short_codes[s.first2() as usize] = FSST_CODE_BASE + self.n_symbols;
        } else if !self.hash_insert(s) {
            return false;
        }
        self.symbols[(FSST_CODE_BASE + self.n_symbols) as usize] = s.clone();
        self.n_symbols += 1;
        self.len_histo[(len - 1) as usize] += 1;
        true
    }

    fn find_longest_symbol_from_char_slice(&self, input: &[u8]) -> u16 {
        let len = if input.len() >= MAX_SYMBOL_LENGTH { MAX_SYMBOL_LENGTH } else { input.len() };
        if len < 2 {
            return self.byte_codes[input[0] as usize] & FSST_CODE_MASK;
        }
        if len == 2 {
            let short_code = (input[1] as usize) << 8 | input[0] as usize;
            if self.short_codes[short_code] & FSST_CODE_UNSET == 0 {
                return self.short_codes[short_code] & FSST_CODE_MASK;
            } else {
                return self.byte_codes[input[0] as usize] & FSST_CODE_MASK;
            }
        }
        let mut input_in_1_word = [0; 8];
        input_in_1_word[..len].copy_from_slice(&input[..len]);
        let input_in_u64 = fsst_unaligned_load_unchecked(input_in_1_word.as_ptr()); 
        let hash_idx = FSST_HASH(input_in_u64) as usize & (self.hash_tab_size - 1);
        let s_in_hash_tab = self.hash_tab[hash_idx];
        if s_in_hash_tab.icl != FSST_ICL_FREE as u64 && s_in_hash_tab.val == (input_in_u64 & (u64::MAX >> s_in_hash_tab.ignored_bits())) {
            return s_in_hash_tab.code();
        }
        return self.byte_codes[input[0] as usize] & FSST_CODE_MASK;
    }

   // rationale for finalize:
   // - during symbol table construction, we may create more than 256 codes, but bring it down to max 255 in the last makeTable()
   //   consequently we needed more than 8 bits during symbol table contruction, but can simplify the codes to single bytes in finalize()
   //   (this feature is in fact lo longer used, but could still be exploited: symbol construction creates no more than 255 symbols in each pass)
   // - we not only reduce the amount of codes to <255, but also *reorder* the symbols and renumber their codes, for higher compression perf.
   //   we renumber codes so they are grouped by length, to allow optimized scalar string compression (byteLim and suffixLim optimizations). 
   // - we make the use of byteCode[] no longer necessary by inserting single-byte codes in the free spots of shortCodes[]
   //   Using shortCodes[] only makes compression faster. When creating the symbolTable, however, using shortCodes[] for the single-byte
   //   symbols is slow, as each insert touches 256 positions in it. This optimization was added when optimizing symbolTable construction time.
   //
   // In all, we change the layout and coding, as follows..
   //
   // before finalize(): 
   // - The real symbols are symbols[256..256+nSymbols>. As we may have nSymbols > 255
   // - The first 256 codes are pseudo symbols (all escaped bytes)
   //
   // after finalize(): 
   // - table layout is symbols[0..nSymbols>, with nSymbols < 256. 
   // - Real codes are [0,nSymbols>. 8-th bit not set. 
   // - Escapes in shortCodes have the 8th bit set (value: 256+255=511). 255 because the code to be emitted is the escape byte 255
   // - symbols are grouped by length: 2,3,4,5,6,7,8, then 1 (single-byte codes last)
   // the two-byte codes are split in two sections: 
   // - first section contains codes for symbols for which there is no longer symbol (no suffix). It allows an early-out during compression
   //
   // finally, shortCodes[] is modified to also encode all single-byte symbols (hence byteCodes[] is not required on a critical path anymore).
   //
    fn finalize(&mut self) {
        assert!(self.n_symbols < FSST_CODE_BASE);
        let mut new_code: [u8; 256] = [0; 256];
        let mut rsum: [u8; 8] = [0; 8];
        let byte_lim = self.n_symbols - self.len_histo[0] as u16;

        rsum[0] = byte_lim as u8; // 1-byte codes are highest
        for i in 1..7 {
            rsum[i + 1] = rsum[i] + self.len_histo[i] as u8;
        }

        let mut suffix_lim = 0;
        let mut j = rsum[2];
        for i in 0..self.n_symbols {
            let mut s1 = self.symbols[(FSST_CODE_BASE + i) as usize];
            let len = s1.length();
            let mut opt = if len == 2 { self.n_symbols } else { 0 };
            if opt != 0 {
                let first2 = s1.first2();
                for k in 0..opt {
                    let s2 = self.symbols[(FSST_CODE_BASE + k)as usize];
                    if k != i && s2.length() > 1 && first2 == s2.first2() {
                        opt = 0;
                    }
                }
                new_code[i as usize] = if opt != 0 { suffix_lim += 1; suffix_lim - 1 } else { j -= 1; j };
            } else {
                new_code[i as usize] = rsum[(len - 1) as usize];
                rsum[(len - 1) as usize] += 1;
            }
            s1.set_code_len(new_code[i as usize] as u16, len);
            self.symbols[new_code[i as usize] as usize] = s1;
        }

        for i in 0..256 {
            if (self.byte_codes[i] & FSST_CODE_MASK) >= FSST_CODE_BASE {
                self.byte_codes[i] = new_code[(self.byte_codes[i] & 0xFF) as usize] as u16;
            } else {
                self.byte_codes[i] = 511;
            }
        }

        for i in 0..65536 {
            if (self.short_codes[i] & FSST_CODE_MASK) > FSST_CODE_BASE {
                self.short_codes[i] = new_code[(self.short_codes[i] & 0xFF) as usize] as u16;
            } else {
                self.short_codes[i] = self.byte_codes[(i & 0xFF) as usize];
            }
        }

        for i in 0..self.hash_tab_size {
            if self.hash_tab[i].icl < FSST_ICL_FREE as u64{
                self.hash_tab[i] = self.symbols[new_code[(self.hash_tab[i].code() & 0xFF) as usize] as usize];
            }
        }
        self.suffix_lim = suffix_lim as u16;
    }
}

#[derive(Clone)]
struct Counters {
    count1: Vec::<u16>,
    count2: Vec::<Vec::<u16>>,
}

impl Counters {
    fn new() -> Self {
        Self {
            count1: vec![0; FSST_CODE_MAX as usize],
            count2: vec![vec![0; FSST_CODE_MAX as usize]; FSST_CODE_MAX as usize],
        }
    }

    fn count1_set(&mut self, pos1: usize, val: u16) {
        self.count1[pos1] = val;
    }

    fn count1_inc(&mut self, pos1: u16) {
        self.count1[pos1 as usize] = self.count1[pos1 as usize].saturating_add(1);
    }

    fn count2_inc(&mut self, pos1: usize, pos2: usize) {
        self.count2[pos1][pos2] = self.count2[pos1][pos2].saturating_add(1);
    }

    fn count1_get(&self, pos1: usize) -> u16 {
        self.count1[pos1]
    }

    fn count2_get(&self, pos1: usize, pos2: usize) -> u16 {
        self.count2[pos1][pos2]
    }

    /* 
    fn clear(&self) {
        Self {
            count1: vec![0; FSST_CODE_MAX as usize],
            count2: vec![vec![0; FSST_CODE_MAX as usize]; FSST_CODE_MAX as usize],
        };
    }*/
}

fn is_escape_code(pos: u16) -> bool {
    pos < FSST_CODE_BASE as u16
}

fn make_sample(strs: &[u8], offsets: &[i32]) -> (Vec<u8>, Vec<i32>) {
    let total_size = strs.len();
    if total_size <= FSST_SAMPLETARGET {
        return (strs.to_vec(), offsets.to_vec());
    }
    let mut sample = Vec::with_capacity(FSST_SAMPLEMAXSZ);
    let mut sample_offsets: Vec<i32> = Vec::new();

    sample_offsets.push(0);
    let mut rng = StdRng::from_entropy();
    while sample.len() < FSST_SAMPLETARGET {
        let rand_num = rng.gen_range(0..offsets.len()) % (offsets.len() - 1);
        sample.extend_from_slice(&strs[offsets[rand_num] as usize ..offsets[rand_num + 1] as usize]);
        sample_offsets.push(sample.len() as i32);
    }
    sample_offsets.push(sample.len() as i32);
    return (sample, sample_offsets);
}

fn build_symbol_table(strs: Vec<u8>, offsets: Vec<i32>) -> io::Result<Box<SymbolTable>> {
    let mut st = SymbolTable::new();
    let mut best_table = SymbolTable::new();
    let mut best_gain = -(FSST_SAMPLEMAXSZ as i32); // worst case (everything exception)

    let mut byte_histo = [0; 256];
    for c in &strs {
        byte_histo[*c as usize] += 1;
    }
   // println!("byte_histo: {:?}", byte_histo);
    let mut curr_min_histo = FSST_SAMPLEMAXSZ;
    
    for i in 0..256 {
        if byte_histo[i] < curr_min_histo {
            curr_min_histo = byte_histo[i];
            st.terminator = i as u16;
        }
    }
    //println!("terminator: {}", st.terminator);
    // Compress sample, and compute (pair-)frequencies
    let compress_count = |st: &mut SymbolTable, sample_frac: usize| -> (Box<Counters>, i32) {
        //println!("symbol table before starting compress_count: {:?}", st);
        // a random number between 1 and 128
        let _rnd128 = |i: usize| -> usize { 
            1 + ((FSST_HASH((i as u64 + 1) * sample_frac as u64)&127) as usize) 
        };
        let mut gain:i32 = 0;
        let mut counters = Counters::new();

        for i in 1..offsets.len() {
            // this is commented out during development
            /* 
            if sample_frac < 128 && _rnd128(i) > sample_frac {
                continue;
            }
            */
            if offsets[i] == offsets[i-1] {
                continue;
            }
            let word = &strs[offsets[i-1] as usize..offsets[i] as usize];

            let mut curr = 0;
            let mut curr_code;
            let mut prev_code = st.find_longest_symbol_from_char_slice(&word[curr..]);
            //println!("prev_code: {}", prev_code);
            curr += st.symbols[prev_code as usize].length() as usize;
            gain += st.symbols[prev_code as usize].length() as i32 - (1 + is_escape_code(prev_code) as i32);
            while curr < word.len() {
                //println!("prev_code: {}, st.symbols[prev_code]: {}", prev_code, st.symbols[prev_code as usize]);
                //https://answers.yahoo.com/question/index?qid=20071007114826AAwCFvR
                counters.count1_inc(prev_code);
                let mut symbol_len = 0;

                if st.symbols[prev_code as usize].length() != 1 {
                    counters.count1_inc(word[curr] as u16);
                }

                if word.len() > 7 && curr < word.len() - 7 {
                    let mut this_64_bit_word: u64 = fsst_unaligned_load_unchecked(word[curr..].as_ptr());
                    let code = this_64_bit_word & 0xFFFFFF;
                    let idx = FSST_HASH(code) as usize & (st.hash_tab_size - 1);
                    let s: Symbol = st.hash_tab[idx];
                    let short_code = st.short_codes[(this_64_bit_word & 0xFFFF) as usize] & FSST_CODE_MASK;
                    this_64_bit_word &= 0xFFFFFFFFFFFFFFFF >> s.icl as u8;
                    if (s.icl < FSST_ICL_FREE as u64) & (s.val == this_64_bit_word) {
                        curr_code = s.code();
                        symbol_len = s.length();
                    } else if short_code >= FSST_CODE_BASE {
                        curr_code = short_code;
                        symbol_len = 2;
                    } else {
                        curr_code = st.byte_codes[(this_64_bit_word & 0xFF) as usize] & FSST_CODE_MASK;
                        symbol_len = 1;
                    }
                } else {
                    curr_code = st.find_longest_symbol_from_char_slice(&word[curr..]);
                    symbol_len = st.symbols[curr_code as usize].length();
                }
                gain += symbol_len as i32 - (1 + is_escape_code(curr_code) as i32);
                if sample_frac < 128 { // no need to count pairs in final round
                    // consider the symbol that is the concatenation of the last two symbols
                    counters.count2_inc(prev_code as usize, curr_code as usize);
                    if symbol_len > 1 {
                        counters.count2_inc(prev_code as usize, word[curr] as usize);
                    }
                }
                curr += symbol_len as usize;
                prev_code = curr_code;
            }
            counters.count1_inc(prev_code);
            /* 
            if st.symbols[prev_code as usize].length() != 1 {
                counters.count1_inc(word[curr - symbol_len as usize] as u16);
            }*/
        }
        //println!("------------------this round finished------------------");
        //println!();
        //println!();
        (Box::new(counters), gain)
    };

    let make_table = |st: &mut SymbolTable, counters: &mut Counters, sample_frac: usize| {
        let mut candidates: HashSet<QSymbol> = HashSet::new();

        counters.count1_set(st.terminator as usize, u16::MAX); 

        let add_or_inc = |cands: &mut HashSet<QSymbol>, s: Symbol, count: u64| {
            if count < (5 * sample_frac as u64) / 128 {
                return;
            }
            let mut q = QSymbol {
                symbol: s,
                gain: (count * s.length() as u64) as u32,
            };
            if let Some(old_q) = cands.get(&q) {
                q.gain += old_q.gain;
                cands.remove(&old_q.clone());
            } 
            cands.insert(q);
        };

        // add candidate symbols based on counted frequencies
        for pos1 in 0..FSST_CODE_BASE as usize + st.n_symbols as usize {
            let cnt1 = counters.count1_get(pos1);
            if cnt1 == 0 {
                continue;
            }
            // heuristic: promoting single-byte symbols (*8) helps reduce exception rates and increases [de]compression speed
            let s1 = st.symbols[pos1];
            add_or_inc(&mut candidates, s1, if s1.length() == 1 { 8 } else { 1 } * cnt1 as u64);
            if s1.first() == st.terminator as u8 {
                continue;
            }
            if sample_frac >= 128 ||
                s1.length() == MAX_SYMBOL_LENGTH as u32 || 
                s1.first() == st.terminator as u8 {
                continue;
            }
            for pos2 in 0..FSST_CODE_BASE as usize + st.n_symbols as usize {
                let cnt2 = counters.count2_get(pos1, pos2);
                if cnt2 == 0 {
                    continue;
                }

                // create a new symbol
                let s2 = st.symbols[pos2];
                let s3 = Symbol::concat(s1, s2);
                // multi-byte symbols cannot contain the terminator byte
                if s2.first() != st.terminator as u8 {
                    /* 
                    println!("s1: {}", s1);
                    println!("s2: {}", s2);
                    println!("s3: {}", s3);
                    println!("new symbol: {}, new_symbol.val: {}", s3, s3.val);
                    */
                    add_or_inc(&mut candidates, s3, cnt2 as u64);
                }
            }
        }
        let mut pq: BinaryHeap<QSymbol> = BinaryHeap::new();
        for q in &candidates {
            pq.push(q.clone());
        }

        // Create new symbol map using best candidates
        st.clear();
        while st.n_symbols < 255 && !pq.is_empty() {
            let q = pq.pop().unwrap();
            st.add(q.symbol);
        }
    };

    for frac in [8, 38, 68, 98, 108, 128] { // we do 5 rounds (sampleFrac=8,38,68,98,128)
    //for frac in [127, 127, 127, 127, 127, 127, 127, 127, 127, 128] {
        let (mut this_counter, gain ) = compress_count(&mut st, frac);
        if gain >= best_gain { // a new best solution!
            best_gain = gain;
            best_table = st.clone();
        } 
        make_table(&mut st, &mut this_counter, frac);
    }
    //println!("before finalize: {}", best_table);
    best_table.finalize(); // renumber codes for more efficient compression
    //println!("after finalize: {}", best_table);
    if best_table.n_symbols == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("Fsst failed to build symbol table, input len: {}, input_offsets len: {}", strs.len(), offsets.len())));
    }
    return Ok(Box::new(best_table));
}    

fn compress_bulk(st: &SymbolTable, strs: &[u8], offsets: &[i32], out: &mut Vec<u8>, out_offsets: &mut Vec<i32>, out_pos: &mut usize, out_offsets_len: &mut usize) -> io::Result<()>{
    //println!("in compress_bulk, st: {}", st);
    let suffix_lim = st.suffix_lim;
    let byte_lim = st.n_symbols - st.len_histo[0] as u16;
    let mut out_curr = *out_pos;

    let mut compress = |buf: &[u8], in_end: usize, out_curr: &mut usize| { 
        let mut in_curr = 0;
        while in_curr < in_end {
            let word = fsst_unaligned_load_unchecked(buf[in_curr..].as_ptr());
            let code = st.short_codes[(word & 0xFFFF) as usize];
            if code < suffix_lim {
                out[*out_curr] = code as u8;
                *out_curr += 1;
                in_curr += 2;
            } else {
                let code_first_3byte= word & 0xFFFFFF;
                let idx = FSST_HASH(code_first_3byte) as usize & (st.hash_tab_size - 1);
                let s = st.hash_tab[idx];
                out[*out_curr + 1] = (word & 0xFF) as u8;
                let word2 = word & (u64::MAX as u64 >> s.ignored_bits());
                if s.icl != FSST_ICL_FREE as u64 && s.val == word2 {
                    //println!("in_curr: {}, *out_curr: {}, hash hit: {}", in_curr, *out_curr, s);
                    out[*out_curr] = s.code() as u8;
                    *out_curr += 1;
                    in_curr += s.length() as usize;
                } else if code < byte_lim {
                    out[*out_curr] = code as u8;
                    *out_curr += 1;
                    in_curr += 2;
                } else {
                    // 1 byte code or miss
                    out[*out_curr] = code as u8;
                    *out_curr += 1 + ((code & 256) >> 8) as usize;
                    in_curr += 1;
                }
            }
        }
    };
    out_offsets[0] = *out_pos as i32;
    for i in 1..offsets.len() {
        let mut in_curr = offsets[i - 1] as usize;
        let end_curr = offsets[i] as usize;
        let mut buf: [u8; 520] = [0; 520]; // +8 sentinel is to avoid 8-byte unaligned-loads going beyond 511 out-of-bounds
        while in_curr < end_curr {
            let in_end = std::cmp::min(in_curr + 511, end_curr);
            {
                let this_len = in_end - in_curr;
                buf[..this_len].copy_from_slice(&strs[in_curr..in_end]);
                buf[this_len] = st.terminator as u8; // sentinel
            }
            compress(&buf, in_end - in_curr, &mut out_curr);
            in_curr = in_end;
        }
        out_offsets[i] = out_curr as i32; 
    }

    out.resize(out_curr, 0); // shrink to actual size
    out_offsets.resize(offsets.len(), 0); // shrink to actual size
    *out_pos = out_curr;
    *out_offsets_len = offsets.len();
    Ok(())
}

fn decompress_bulk(st: &SymbolTable, compressed_strs: &[u8], offsets: &[i32], out: &mut Vec<u8>, out_offsets: &mut Vec<i32>, out_pos: &mut usize, out_offsets_len: &mut usize) -> io::Result<()> {
    let mut decompress = |mut in_curr: usize, in_end: usize, out_curr: &mut usize| {
        let mut prev_esc = false;
        while in_curr < in_end {
            if prev_esc {
                out[*out_curr] = compressed_strs[in_curr];
                *out_curr += 1;
                prev_esc = false;
            } else {
                let code = compressed_strs[in_curr];
                if code == FSST_ESC {
                    prev_esc = true;
                } else {
                    let s = st.symbols[code as usize];
                    let len = s.length();
                    out[*out_curr..*out_curr + len as usize].copy_from_slice(&s.val.to_ne_bytes()[..len as usize]);
                    *out_curr += len as usize;
                }
            }
            in_curr += 1;
        }
    };
    let mut out_curr = *out_pos;
    out_offsets[0] = 0;
    for i in 1..offsets.len() {
        let in_curr = offsets[i - 1] as usize;
        let in_end = offsets[i] as usize;
        decompress(in_curr, in_end, &mut out_curr);
        out_offsets[i] = out_curr as i32;
    }
    //println!("out: {:?}", out);
    out.resize(out_curr, 0);
    out_offsets.resize(offsets.len(), 0);
    *out_pos = out_curr;
    *out_offsets_len = offsets.len();
    Ok(())
}

fn decompress_bulk2(symbols: &[Symbol], compressed_strs: &[u8], offsets: &[i32], _in_curr: &mut usize, out: &mut Vec<u8>, out_offsets: &mut Vec<i32>, out_pos: &mut usize, out_offsets_len: &mut usize) -> io::Result<()> {
    let mut decompress = |mut in_curr: usize, in_end: usize, out_curr: &mut usize| {
        let mut prev_esc = false;
        while in_curr < in_end {
            if prev_esc {
                out[*out_curr] = compressed_strs[in_curr];
                *out_curr += 1;
                prev_esc = false;
            } else {
                let code = compressed_strs[in_curr];
                if code == FSST_ESC {
                    prev_esc = true;
                } else {
                    let s = symbols[code as usize];
                    let len = s.length();
                    out[*out_curr..*out_curr + len as usize].copy_from_slice(&s.val.to_ne_bytes()[..len as usize]);
                    *out_curr += len as usize;
                }
            }
            in_curr += 1;
        }
    };
    let mut out_curr = *out_pos;
    out_offsets[0] = 0;
    for i in 1..offsets.len() {
        let in_curr = offsets[i - 1] as usize;
        let in_end = offsets[i] as usize;
        decompress(in_curr, in_end, &mut out_curr);
        out_offsets[i] = out_curr as i32;
    }
    out.resize(out_curr, 0);
    out_offsets.resize(offsets.len(), 0);
    *out_pos = out_curr;
    *out_offsets_len = offsets.len();
    Ok(())
}

// use a struct so we can have many implementations based on cpu type
struct FsstEncoder {
    symbol_table: Box<SymbolTable>,
    // when in_buf is less than FSST_LEAST_INPUT_SIZE, we simply copy the input to the output 
    encoder_switch: bool, 
}

impl FsstEncoder {
    fn new() -> Self {
        Self {
            symbol_table: Box::new(SymbolTable::new()),
            encoder_switch: false,
        }
    }

    fn init(&mut self, in_buf: &[u8], in_offsets_buf: &[i32], out_buf: &Vec<u8>, _out_offsets_buf: &Vec<i32>) -> io::Result<()> {
        if in_buf.len() < FSST_LEAST_INPUT_SIZE {
            return Ok(());
        }

        // currently, we make sure the compress output buffer has the same size as the input buffer,
        // because I don't know a good way to estimate this yet
        if in_buf.len() > out_buf.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "output buffer too small for FSST encoder"));
        }

        self.encoder_switch = true;
        let (sample, sample_offsets) = make_sample(in_buf, in_offsets_buf);
        let st = build_symbol_table(sample, sample_offsets)?;
        self.symbol_table = st;
        Ok(())
    }

    fn export(&self, out_buf: &mut Vec<u8>, out_pos: &mut usize) -> io::Result<()> {
        let st = &self.symbol_table;

        let st_info: u64 = FSST_MAGIC |  
                            ((st.suffix_lim & FSST_CODE_BASE) as u64) << 16 |
                            ((st.terminator & FSST_CODE_BASE) as u64) << 8 | 
                            ((st.n_symbols & FSST_CODE_BASE) as u64);

        let st_info_bytes = st_info.to_ne_bytes();
        out_buf[*out_pos..*out_pos + st_info_bytes.len()].copy_from_slice(&st_info_bytes);
        
        *out_pos += st_info_bytes.len();
        for i in 0..8 {
            out_buf[*out_pos] = st.len_histo[i];
            *out_pos += 1;
        }

        for i in 0..st.n_symbols as usize {
            let s = st.symbols[i];
            let s_bytes = s.val.to_ne_bytes();
            out_buf[*out_pos..*out_pos + s_bytes.len()].copy_from_slice(&s_bytes);
            *out_pos += s_bytes.len();
            let s_icl = s.icl.to_ne_bytes();
            out_buf[*out_pos..*out_pos + s_icl.len()].copy_from_slice(&s_icl);
            *out_pos += s_icl.len();
        }
        Ok(())
    }


    fn compress(&mut self, in_buf: &[u8], in_offsets_buf: &[i32], out_buf: &mut Vec<u8>, out_offsets_buf: &mut Vec<i32>) -> io::Result<()> {
        self.init(&in_buf, &in_offsets_buf, &out_buf, &out_offsets_buf)?;

        // if the input buffer is less than FSST_LEAST_INPUT_SIZE, we simply copy the input to the output
        if self.encoder_switch == false {
            out_buf.resize(in_buf.len(), 0);
            out_buf.copy_from_slice(in_buf);
            out_offsets_buf.resize(in_offsets_buf.len(), 0);
            out_offsets_buf.copy_from_slice(in_offsets_buf);
            return Ok(());
        }
        let mut out_pos = 0;
        self.export(out_buf, &mut out_pos)?;
        let mut out_offsets_len = 0;
        compress_bulk(&self.symbol_table, in_buf, in_offsets_buf, out_buf, out_offsets_buf, &mut out_pos, &mut out_offsets_len)?;
        Ok(())
    }
}

// use a struct so we can have many implementations based on cpu type
struct FsstDecoder {
    len_histo: [u8; 8],
    symbols: [Symbol; 256],
    decoder_switch: bool,
}

const FSST_CORRUPT: u64 = 32774747032022883; // 7-byte number in little endian containing "corrupt"
impl FsstDecoder {
    fn new() -> Self {
        let s = Symbol::new();
        Self {
            len_histo: [0; 8],
            symbols: [s; 256],
            decoder_switch: false,
        }
    }
    fn init(&mut self, in_buf: &[u8], in_pos: &mut usize, out_buf: &Vec<u8>, _out_offsets_buf: &Vec<i32>) -> io::Result<()> {
        if in_buf.len() < FSST_LEAST_INPUT_SIZE {
            return Ok(());
        }

        // currently, we make sure the out_buf is at least 3 times the size of the in_buf, 
        // because I don't know a good way to estimate this
        if in_buf.len() * 3 > out_buf.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "output buffer too small for FSST decoder"));
        }
        self.decoder_switch = true;
        let st_info = u64::from_ne_bytes(in_buf[*in_pos..*in_pos + 8].try_into().unwrap());
        if st_info & FSST_MAGIC != FSST_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "the input buffer is not a valid FSST compressed data"));
        }
        *in_pos += 8;
        let mut len_histo = [0; 8];
        len_histo.copy_from_slice(&in_buf[*in_pos..*in_pos + 8]);
        let mut symbols = [Symbol::new(); 256];
        *in_pos += len_histo.len();
        let mut code = 0;
        for i in [2, 3, 4, 5, 6, 7, 8, 1] {
            let this_len_histo = len_histo[i - 1];
            for _ in 0..this_len_histo {
                symbols[code].val = fsst_unaligned_load_unchecked(in_buf[*in_pos..].as_ptr());
                *in_pos += 8;
                symbols[code].icl = fsst_unaligned_load_unchecked(in_buf[*in_pos..].as_ptr());
                *in_pos += 8;
                code += 1;
            }
        }
        for curr_code in code..256 {
            symbols[curr_code].val = FSST_CORRUPT;
        }
        self.len_histo = len_histo;
        self.symbols = symbols;
        Ok(())
    }

    fn decompress(&mut self, in_buf: &[u8], in_offsets_buf: &[i32], out_buf: &mut Vec<u8>, out_offsets_buf: &mut Vec<i32>) -> io::Result<()> {
        let mut in_pos = 0;
        self.init(in_buf, &mut in_pos, &out_buf, &out_offsets_buf)?;

        if self.decoder_switch == false {
            out_buf.resize(in_buf.len(), 0);
            out_buf.copy_from_slice(in_buf);
            out_offsets_buf.resize(in_offsets_buf.len(), 0);
            out_offsets_buf.copy_from_slice(in_offsets_buf);
            return Ok(());
        }

        let mut out_pos = 0;
        let mut out_offsets_len = 0;
        decompress_bulk2(&self.symbols, in_buf, in_offsets_buf, &mut in_pos, out_buf, out_offsets_buf, &mut out_pos, &mut out_offsets_len)?;
        Ok(())
    }
}


pub fn compress(in_buf: &[u8], in_offsets_buf: &[i32], out_buf: &mut Vec<u8>, out_offsets_buf: &mut Vec<i32>) -> io::Result<()> {
    FsstEncoder::new().compress(in_buf, in_offsets_buf, out_buf, out_offsets_buf)?;
    Ok(())
}

pub fn decompress(in_buf: &[u8], in_offsets_buf: &[i32], out_buf: &mut Vec<u8>, out_offsets_buf: &mut Vec<i32>) -> io::Result<()> {
    FsstDecoder::new().decompress(in_buf, in_offsets_buf, out_buf, out_offsets_buf)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use arrow::array::StringArray;
    use rand::Rng;

    use crate::fsst::*;

    const TEST_PARAGRAPH: &str = "ACT I. Scene I.
    Elsinore. A platform before the Castle.
    
    Enter two Sentinels-[first,] Francisco, [who paces up and down
    at his post; then] Bernardo, [who approaches him].
    
        Ber. Who's there.?
        Fran. Nay, answer me. Stand and unfold yourself.
        Ber. Long live the King!
        Fran. Bernardo?
        Ber. He.
        Fran. You come most carefully upon your hour.
        Ber. 'Tis now struck twelve. Get thee to bed, Francisco.
        Fran. For this relief much thanks. 'Tis bitter cold,
        And I am sick at heart.
        Ber. Have you had quiet guard?
        Fran. Not a mouse stirring.
        Ber. Well, good night.
        If you do meet Horatio and Marcellus,
        The rivals of my watch, bid them make haste.
        Enter Horatio and Marcellus.  

        Fran. I think I hear them. Stand, ho! Who is there?
        Hor. Friends to this ground.
        Mar. And liegemen to the Dane.
        Fran. Give you good night.
        Mar. O, farewell, honest soldier.
            Who hath reliev'd you?
        Fran. Bernardo hath my place.
            Give you good night.                                   Exit.
        Mar. Holla, Bernardo!
        Ber. Say-
            What, is Horatio there ?
        Hor. A piece of him.
        Ber. Welcome, Horatio. Welcome, good Marcellus.
        Mar. What, has this thing appear'd again to-night?
        Ber. I have seen nothing.
        Mar. Horatio says 'tis but our fantasy,
            And will not let belief take hold of him
            Touching this dreaded sight, twice seen of us.
            Therefore I have entreated him along,  
            With us to watch the minutes of this night,
            That, if again this apparition come,
            He may approve our eyes and speak to it.
        Hor. Tush, tush, 'twill not appear.
        Ber. Sit down awhile,
            And let us once again assail your ears,
            That are so fortified against our story,
            What we two nights have seen.
        Hor. Well, sit we down,
            And let us hear Bernardo speak of this.
        Ber. Last night of all,
            When yond same star that's westward from the pole
            Had made his course t' illume that part of heaven
            Where now it burns, Marcellus and myself,
            The bell then beating one-

                                Enter Ghost.

        Mar. Peace! break thee off! Look where it comes again!
        Ber. In the same figure, like the King that's dead.  
        Mar. Thou art a scholar; speak to it, Horatio.
        Ber. Looks it not like the King? Mark it, Horatio.
        Hor. Most like. It harrows me with fear and wonder.
        Ber. It would be spoke to.
        Mar. Question it, Horatio.
        Hor. What art thou that usurp'st this time of night
            Together with that fair and warlike form
            In which the majesty of buried Denmark
            Did sometimes march? By heaven I charge thee speak!
        Mar. It is offended.
        Ber. See, it stalks away!
        Hor. Stay! Speak, speak! I charge thee speak!
                                                            Exit Ghost.
        Mar. 'Tis gone and will not answer.
        Ber. How now, Horatio? You tremble and look pale.
            Is not this something more than fantasy?
            What think you on't?
        Hor. Before my God, I might not this believe
            Without the sensible and true avouch
            Of mine own eyes.  
        Mar. Is it not like the King?
        Hor. As thou art to thyself.
            Such was the very armour he had on
            When he th' ambitious Norway combated.
            So frown'd he once when, in an angry parle,
            He smote the sledded Polacks on the ice.
            'Tis strange.
        Mar. Thus twice before, and jump at this dead hour,
            With martial stalk hath he gone by our watch.
        Hor. In what particular thought to work I know not;
            But, in the gross and scope of my opinion,
            This bodes some strange eruption to our state.
        Mar. Good now, sit down, and tell me he that knows,
            Why this same strict and most observant watch
            So nightly toils the subject of the land,
            And why such daily cast of brazen cannon
            And foreign mart for implements of war;
            Why such impress of shipwrights, whose sore task
            Does not divide the Sunday from the week.
            What might be toward, that this sweaty haste  
            Doth make the night joint-labourer with the day?
            Who is't that can inform me?
        Hor. That can I.
            At least, the whisper goes so. Our last king,
            Whose image even but now appear'd to us,
            Was, as you know, by Fortinbras of Norway,
            Thereto prick'd on by a most emulate pride,
            Dar'd to the combat; in which our valiant Hamlet
            (For so this side of our known world esteem'd him)
            Did slay this Fortinbras; who, by a seal'd compact,
            Well ratified by law and heraldry,
            Did forfeit, with his life, all those his lands
            Which he stood seiz'd of, to the conqueror;
            Against the which a moiety competent
            Was gaged by our king; which had return'd
            To the inheritance of Fortinbras,
            Had he been vanquisher, as, by the same comart
            And carriage of the article design'd,
            His fell to Hamlet. Now, sir, young Fortinbras,
            Of unimproved mettle hot and full,  
            Hath in the skirts of Norway, here and there,
            Shark'd up a list of lawless resolutes,
            For food and diet, to some enterprise
            That hath a stomach in't; which is no other,
            As it doth well appear unto our state,
            But to recover of us, by strong hand
            And terms compulsatory, those foresaid lands
            So by his father lost; and this, I take it,
            Is the main motive of our preparations,
            The source of this our watch, and the chief head
            Of this post-haste and romage in the land.
        Ber. I think it be no other but e'en so.
            Well may it sort that this portentous figure
            Comes armed through our watch, so like the King
            That was and is the question of these wars.
        Hor. A mote it is to trouble the mind's eye.
            In the most high and palmy state of Rome,
            A little ere the mightiest Julius fell,
            The graves stood tenantless, and the sheeted dead
            Did squeak and gibber in the Roman streets;  
            As stars with trains of fire, and dews of blood,
            Disasters in the sun; and the moist star
            Upon whose influence Neptune's empire stands
            Was sick almost to doomsday with eclipse.
            And even the like precurse of fierce events,
            As harbingers preceding still the fates
            And prologue to the omen coming on,
            Have heaven and earth together demonstrated
            Unto our climature and countrymen.";

    #[test_log::test(tokio::test)]
    async fn test_symbol_new() {
        let st = SymbolTable::new();
        assert!(st.n_symbols == 0);
        for i in 0..=255 as u8 {
            assert!(st.symbols[i as usize] == Symbol::from_char(i, i as u16));
        }
        let s = Symbol::from_char(1, 1);
        assert!(s == st.symbols[1]);
        for i in 0..1 << FSST_HASH_LOG2SIZE {
            assert!(st.hash_tab[i] == Symbol::new());
        }
    }

    #[test_log::test(tokio::test)]
    async fn test_symbol_from_slice() {
        let hello_str = "hello";
        let symbol_hello = Symbol::from_char_slice(hello_str.as_bytes());
        assert!(symbol_hello.length() == hello_str.len() as u32);
        assert!(symbol_hello.ignored_bits() == 24); // 8 - 5 = 3
        for i in 0..hello_str.len() {
            assert!(symbol_hello.val.to_ne_bytes()[i] == hello_str.as_bytes()[i]);
        }
    }

    #[test_log::test(tokio::test)]
    async fn test_symbol_add() {
        let mut st = SymbolTable::new();
        let hello_str = "hello";
        st.add(Symbol::from_char_slice(hello_str.as_bytes()));
        let symbol_hello = Symbol::from_char_slice(hello_str.as_bytes());
        assert!(st.symbols[FSST_CODE_BASE as usize].length() == symbol_hello.length());
        assert!(st.symbols[FSST_CODE_BASE as usize].ignored_bits() == symbol_hello.ignored_bits());
        assert!(st.symbols[FSST_CODE_BASE as usize].val == symbol_hello.val);
        assert!(st.n_symbols == 1);
        assert!(false == st.add(symbol_hello));
        assert!(st.n_symbols == 1);
        let world_str = "world";
        let symbol_world = Symbol::from_char_slice(world_str.as_bytes());
        st.add(symbol_world);
        assert!(st.n_symbols == 2);
        assert!(st.symbols[FSST_CODE_BASE as usize + 1].length() == symbol_world.length());
        assert!(st.symbols[FSST_CODE_BASE as usize + 1].ignored_bits() == symbol_world.ignored_bits());
        assert!(st.symbols[FSST_CODE_BASE as usize + 1].val == symbol_world.val);
        let us_str = "us";
        let us_symbol = Symbol::from_char_slice(us_str.as_bytes());
        st.add(us_symbol);
        assert!(st.n_symbols == 3);
        assert!(st.symbols[FSST_CODE_BASE as usize + 2].val == us_symbol.val);
        assert!(st.symbols[FSST_CODE_BASE as usize + 2].ignored_bits() == us_symbol.ignored_bits());
        assert!(st.symbols[FSST_CODE_BASE as usize + 2].length() == us_symbol.length());
        let short_codes_idx: usize = u16::from_ne_bytes([us_str.as_bytes()[0], us_str.as_bytes()[1]]) as usize;
        let code_in_short_codes = st.short_codes[short_codes_idx];
        assert!(code_in_short_codes as usize == (FSST_CODE_BASE + 2) as usize);
        let x_str = "x";
        let x_symbol = Symbol::from_char_slice(x_str.as_bytes());
        st.add(x_symbol);
        assert!(st.n_symbols == 4);
        assert!(st.symbols[FSST_CODE_BASE as usize + 3].val == x_symbol.val);
        assert!(st.symbols[FSST_CODE_BASE as usize + 3].ignored_bits() == x_symbol.ignored_bits());
        assert!(st.symbols[FSST_CODE_BASE as usize + 3].length() == x_symbol.length());
        assert!(st.symbols[code_in_short_codes as usize].val == us_symbol.val);
    }

    // to run this test, download MS Marco dataset from https://msmarco.z22.web.core.windows.net/msmarcoranking/fulldocs.tsv.gz
    // and use a script like this to get each column, then uncomment the block in test_make_sample
    /* 
    import csv
    import sys

    def write_second_column(input_path, output_path):
        csv.field_size_limit(sys.maxsize)

        with open(input_path, 'r') as input_file, open(output_path, 'w') as output_file:
            tsv_reader = csv.reader(input_file, delimiter='\t')
            tsv_writer = csv.writer(output_file, delimiter='\t')

            for row in tsv_reader:
                tsv_writer.writerow([row[2]])

    #write_second_column('/Users/x/fulldocs.tsv', '/Users/x/first_column_fulldocs.tsv')
    #write_second_column('/Users/x/fulldocs.tsv', '/Users/x/second_column_fulldocs.tsv')
    write_second_column('/Users/x/fulldocs.tsv', '/Users/x/third_column_fulldocs.tsv')
    */
    #[test_log::test(tokio::test)]
    async fn test_make_sample() {
        /* 
        let file_paths = [
            //"/Users/x/first_column_fulldocs.tsv",
            "/Users/x/second_column_fulldocs.tsv",
            //"/Users/x/third_column_fulldocs.tsv",
        ];
        for file_path in file_paths {
            let input = read_random_16_m_chunk(file_path).unwrap();
            let (sample_input, sample_offsets) = make_sample(input.values(), input.value_offsets());
            println!("first sample string {:?}", std::str::from_utf8(&sample_input[sample_offsets[0] as usize..sample_offsets[1] as usize]));
            println!("sample size: {}", sample_input.len());
            println!("sample string number: {}", sample_offsets.len() - 1);
        }
        */
    }

    // to run this test, download MS Marco dataset from https://msmarco.z22.web.core.windows.net/msmarcoranking/fulldocs.tsv.gz
    // and use a script like this to get each column, then uncomment the block in test_build_symbol_table
    /* 
    import csv
    import sys

    def write_second_column(input_path, output_path):
        csv.field_size_limit(sys.maxsize)

        with open(input_path, 'r') as input_file, open(output_path, 'w') as output_file:
            tsv_reader = csv.reader(input_file, delimiter='\t')
            tsv_writer = csv.writer(output_file, delimiter='\t')

            for row in tsv_reader:
                tsv_writer.writerow([row[2]])

    #write_second_column('/Users/x/fulldocs.tsv', '/Users/x/first_column_fulldocs.tsv')
    #write_second_column('/Users/x/fulldocs.tsv', '/Users/x/second_column_fulldocs.tsv')
    write_second_column('/Users/x/fulldocs.tsv', '/Users/x/third_column_fulldocs.tsv')
    */
    #[test_log::test(tokio::test)]
    async fn test_build_symbol_table() {
        /* 
        let paragraph = TEST_PARAGRAPH.to_string();
        let words = paragraph.lines().collect::<Vec<&str>>();
        let string_array = StringArray::from(words);
        let st = *build_symbol_table(string_array.value_data().to_vec(), string_array.value_offsets().to_vec()).unwrap();
        println!("{}", st);
        */
        // to test build_symbol_table, uncomment this block
        let file_paths = [
            "/home/x/first_column_fulldocs.tsv",
            //"/home/x/second_column_fulldocs.tsv",
            //"/home/x/third_column_fulldocs.tsv",
        ];
        for file_path in file_paths {
            let input = read_random_16_m_chunk(file_path).unwrap();
            let (sample_input, sample_offsets) = make_sample(input.values(), input.value_offsets());
            let st = *build_symbol_table(sample_input, sample_offsets).unwrap();
            println!("{}", st);
        }
    }

    #[test_log::test(tokio::test)]
    async fn test_compress_bulk() {

        let paragraph = TEST_PARAGRAPH.to_string();
        let words = paragraph.lines().collect::<Vec<&str>>();
        let string_array = StringArray::from(words);
        let (sample_input, sample_offsets) = make_sample(string_array.values(), string_array.value_offsets());
        let st = *build_symbol_table(sample_input, sample_offsets).unwrap();
        let mut compress_output_buf: Vec<u8> = vec![0; 16 * 1024 * 1024];
        let mut compress_offset_buf: Vec<i32> = vec![0; 16 * 1024 * 1024];
        let mut compress_out_buf_pos = 0;
        let mut compress_out_offsets_len = 0;
        compress_bulk(&st, string_array.values(), string_array.value_offsets(), &mut compress_output_buf, &mut compress_offset_buf, &mut compress_out_buf_pos, &mut compress_out_offsets_len).unwrap();
        let this_compression_ratio = string_array.values().len() as f64 / compress_out_buf_pos as f64;
        println!("compression ratio: {}", this_compression_ratio);
        // due to the non-deterministic nature in the sampling phase, I couldn't find the good way to test this,
        // we can print out the symbol table and parts of the compress_output_buf to inspect 
        println!("symbol table: {}", st);
        println!("PARAGRAPH[0.100]: {}", TEST_PARAGRAPH[0..100].to_string());
        println!("compress_output_buf[0..100]: {:?}", &compress_output_buf[0..100]);
    }

    // to run this test, download MS Marco dataset from https://msmarco.z22.web.core.windows.net/msmarcoranking/fulldocs.tsv.gz
    // and use a script like this to get each column
    /* 
    import csv
    import sys

    def write_second_column(input_path, output_path):
        csv.field_size_limit(sys.maxsize)

        with open(input_path, 'r') as input_file, open(output_path, 'w') as output_file:
            tsv_reader = csv.reader(input_file, delimiter='\t')
            tsv_writer = csv.writer(output_file, delimiter='\t')

            for row in tsv_reader:
                tsv_writer.writerow([row[2]])

    #write_second_column('/Users/x/fulldocs.tsv', '/Users/x/first_column_fulldocs.tsv')
    #write_second_column('/Users/x/fulldocs.tsv', '/Users/x/second_column_fulldocs.tsv')
    write_second_column('/Users/x/fulldocs.tsv', '/Users/x/third_column_fulldocs.tsv')
    */
    // 
    #[test_log::test(tokio::test)]
    async fn test_fsst_without_seralize_symbol_table() {
        for _ in 0..1 {
            let num = rand::thread_rng().gen_range(1..10);
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(num); 
            let mut generator = lance_datagen::array::rand_utf8(lance_datagen::ByteCount::from(num), false);
            let result = generator.generate((8 * 1024 * 1024).into(), &mut rng).unwrap(); // so we generate 16MB * num of data
            let string_array = result.as_any().downcast_ref::<StringArray>().unwrap();
            let (sample_strs_buffer, sample_offsets_buffer) = make_sample(string_array.values(), string_array.value_offsets());
            let st = *build_symbol_table(sample_strs_buffer, sample_offsets_buffer).unwrap();
            // in the case of randomly generated data, we expect the compression ratio to be low, so we allocate 2 * input size here
            let mut compress_output_buffer: Vec<u8> = vec![0; 2 * 8 * 1024 * 1024 * num as usize]; 
            let mut compress_offset_buffer: Vec<i32> = vec![0; 2 * 8 * 1024 * 1024 * num as usize];
            let mut compress_out_buf_pos = 0;
            let mut compress_out_offsets_len = 0;
            compress_bulk(&st, string_array.values(), string_array.value_offsets(), &mut compress_output_buffer, &mut compress_offset_buffer, &mut compress_out_buf_pos, & mut compress_out_offsets_len).unwrap();
            let mut decompressed_output: Vec<u8> = vec![0; 3 * 2 * 8 * 1024 * 1024 * num as usize];
            let mut decompressed_offsets: Vec<i32> = vec![0; 3 * 2 * 8 * 1024 * 1024 * num as usize];
            let mut decompressed_output_pos = 0;
            let mut decompressed_offsets_len = 0;
            let _ = decompress_bulk(&st, &compress_output_buffer, &compress_offset_buffer, &mut decompressed_output, &mut decompressed_offsets, &mut decompressed_output_pos, &mut decompressed_offsets_len);
            assert!(decompressed_offsets_len == string_array.value_offsets().to_vec().len());
            assert!(decompressed_output_pos == string_array.values().len());
            for i in 1..decompressed_offsets_len {
                let s = &decompressed_output[decompressed_offsets[i-1] as usize..decompressed_offsets[i] as usize];
                let original = &string_array.value_data()[string_array.value_offsets().to_vec()[i-1] as usize..string_array.value_offsets().to_vec()[i] as usize];
                assert!(s == original);
            }
        }

        /* 
        let paragraph2 = TEST_PARAGRAPH.to_string().repeat(1024);
        let words = paragraph2.split_whitespace().collect::<Vec<&str>>();
        let string_array = StringArray::from(words);
        let st = *build_symbol_table(string_array.value_data().to_vec(), string_array.value_offsets().to_vec()).unwrap();
        let mut compress_output_buffer: Vec<u8> = vec![0; 16 * 1024 * 1024];
        let mut compress_offset_buffer: Vec<i32> = vec![0; 16 * 1024 * 1024];
        let mut compress_out_buf_pos = 0;
        let mut compress_out_offsets_len = 0;
        compress_bulk(&st, string_array.value_data(), string_array.value_offsets(), &mut compress_output_buffer, &mut compress_offset_buffer, &mut compress_out_buf_pos, &mut compress_out_offsets_len).unwrap();
        println!("compress_out_buf_pos: {:?}", compress_out_buf_pos);
        println!("string_array.values().len(): {:?}", string_array.values().len());
        println!("compression_ratio: {:?}", compress_out_buf_pos as f64 / string_array.values().len() as f64);
        let mut decompressed_output: Vec<u8> = vec![0; 3 * 16 * 1024 * 1024];
        let mut decompressed_offsets: Vec<i32> = vec![0; 3 * 16 * 1024 * 1024];
        let mut decompressed_output_pos = 0;
        let mut decompressed_offsets_len = 0;
        let _ = decompress_bulk(&st, &compress_output_buffer, &compress_offset_buffer, &mut decompressed_output, &mut decompressed_offsets, &mut decompressed_output_pos, &mut decompressed_offsets_len);
        for i in 1..string_array.value_offsets().to_vec().len() {
            let s = &decompressed_output[decompressed_offsets[i-1] as usize..decompressed_offsets[i] as usize];
            let original = &string_array.value_data()[string_array.value_offsets().to_vec()[i-1] as usize..string_array.value_offsets().to_vec()[i] as usize];
            assert!(s == original);
        }
        */
        // to run this test, download MS Marco dataset from https://msmarco.z22.web.core.windows.net/msmarcoranking/fulldocs.tsv.gz
        // and use a script like this to get each column
        /* 
        import csv
        import sys

        def write_second_column(input_path, output_path):
            csv.field_size_limit(sys.maxsize)

            with open(input_path, 'r') as input_file, open(output_path, 'w') as output_file:
                tsv_reader = csv.reader(input_file, delimiter='\t')
                tsv_writer = csv.writer(output_file, delimiter='\t')

                for row in tsv_reader:
                    tsv_writer.writerow([row[2]])

        #write_second_column('/Users/x/fulldocs.tsv', '/Users/x/first_column_fulldocs.tsv')
        #write_second_column('/Users/x/fulldocs.tsv', '/Users/x/second_column_fulldocs.tsv')
        write_second_column('/Users/x/fulldocs.tsv', '/Users/x/third_column_fulldocs.tsv')
        */
        /* 
        let test_num = 1;
        let file_paths = [
            "/Users/x/first_column_fulldocs.tsv",
            "/Users/x/second_column_fulldocs.tsv",
            "/Users/x/third_column_fulldocs.tsv",
        ];
        for file_path in file_paths {
            let mut compression_ratio_sum: f64 = 0.0;
            for _ in 0..test_num {
                let input = read_random_16_m_chunk(file_path).unwrap();
                let (sample_input, sample_offsets) = make_sample(input.values(), input.value_offsets());
                let st = *build_symbol_table(sample_input, sample_offsets).unwrap();
                //println!("symbol table: {}", st);
                let mut compress_output_buffer: Vec<u8> = vec![0; 16 * 1024 * 1024 * 2]; // 16MB * 2
                let mut compress_offset_buffer: Vec<i32> = vec![0; 16 * 1024 * 1024 * 2];
                let mut compress_out_buf_pos = 0;
                let mut compress_out_offsets_len = 0;
                compress_bulk(&st, input.values(), input.value_offsets(), &mut compress_output_buffer, &mut compress_offset_buffer, &mut compress_out_buf_pos, &mut compress_out_offsets_len).unwrap();
                assert!(compress_out_offsets_len == input.value_offsets().to_vec().len());
                //println!("compress_out_buf_pos: {:?}", compress_out_buf_pos);
                //println!("input.values().len(): {:?}", input.values().len());
                let this_compression_ratio = input.values().len() as f64 / compress_out_buf_pos as f64;
                println!("this_compression_ratio: {:?}", this_compression_ratio);
                compression_ratio_sum += this_compression_ratio;
                let mut decompressed_output: Vec<u8> = vec![0; 16 * 1024 * 1024 * 2];
                let mut decompressed_offsets: Vec<i32> = vec![0; 16 * 1024 * 1024 * 2];
                let mut decompressed_output_pos = 0;
                let mut decompressed_offsets_len = 0;
                let mut _in_pos = 0;
                let _ = decompress_bulk(&st, &compress_output_buffer, &compress_offset_buffer, &mut decompressed_output, &mut decompressed_offsets, &mut decompressed_output_pos, &mut decompressed_offsets_len);
                //println!("decompressed_output_pos: {:?}", decompressed_output_pos);
                //println!("decompressed_offsets_len: {:?}", decompressed_offsets_len);
                //println!("input.values().len(): {:?}", input.values().len());
                assert!(decompressed_offsets_len == input.value_offsets().to_vec().len());
                assert!(decompressed_output_pos == input.values().len());
                for i in 1..decompressed_offsets_len {
                    let s = &decompressed_output[decompressed_offsets[i-1] as usize..decompressed_offsets[i] as usize];
                    let original = &input.value_data()[input.value_offsets().to_vec()[i-1] as usize..input.value_offsets().to_vec()[i] as usize];
                    //println!("s: {:?}", std::str::from_utf8(s));
                    //println!("original: {:?}", std::str::from_utf8(original));
                    assert!(s == original);
                }
            }
            println!("for file: {}, average compression_ratio: {:?}", file_path, compression_ratio_sum / test_num as f64);
        }
        */
    }

    #[test_log::test(tokio::test)]
    async fn test_fsst() {
        let paragraph = TEST_PARAGRAPH.to_string().repeat(1024);
        let words = paragraph.lines().collect::<Vec<&str>>();
        let string_array = StringArray::from(words);
        let mut compress_output_buf: Vec<u8> = vec![0; 16 * 1024 * 1024];
        let mut compress_offset_buf: Vec<i32> = vec![0; 16 * 1024 * 1024];
        compress(string_array.value_data(), string_array.value_offsets(), &mut compress_output_buf, &mut compress_offset_buf).unwrap();
        let this_compression_ratio = string_array.value_data().len() as f64 / compress_output_buf.len() as f64;
        println!("this_compression_ratio: {:?}", this_compression_ratio);
        let mut decompress_output: Vec<u8> = vec![0; 3 * 16 * 1024 * 1024];
        let mut decompress_offsets: Vec<i32> = vec![0; 3 * 16 * 1024 * 1024];
        decompress(&compress_output_buf, &compress_offset_buf, &mut decompress_output, &mut decompress_offsets).unwrap();
        println!("decompress_offsets.len(): {}", decompress_offsets.len());
        for i in 1..decompress_offsets.len() {
            let s = &decompress_output[decompress_offsets[i-1] as usize..decompress_offsets[i] as usize];
            let original = &string_array.value_data()[string_array.value_offsets().to_vec()[i-1] as usize..string_array.value_offsets().to_vec()[i] as usize];
            //println!("s: {:?}", std::str::from_utf8(s));
            //println!("original: {:?}", std::str::from_utf8(original));
            assert!(s == original, "s: {:?}\n\n, original: {:?}", std::str::from_utf8(s), std::str::from_utf8(original));
        }
        /* 
        let test_num = 1;
        let file_paths = [
            //"/Users/x/first_column_fulldocs.tsv",
            "/home/x/second_column_fulldocs.tsv",
            //"/Users/x/third_column_fulldocs.tsv",
        ];
        for file_path in file_paths {
            let mut compression_ratio_sum: f64 = 0.0;
            for _ in 0..test_num {
                let input = read_random_32_m_chunk(file_path).unwrap();
                let mut encoder = FsstEncoder::new();
                let mut compress_output_buffer: Vec<u8> = vec![0; 32 * 1024 * 1024 + 8096]; 
                let mut compress_offset_buffer: Vec<i32> = vec![0; 32 * 1024 * 1024 + 8096];
                encoder.compress(input.value_data(), input.value_offsets(), &mut compress_output_buffer, & mut compress_offset_buffer).unwrap();
                //println!("compress_output_buffer.len(): {}", compress_output_buffer.len());
                //println!("compree_offset_buffer.len(): {}", compress_offset_buffer.len());
                let this_compression_ratio = input.value_data().len() as f64 / compress_output_buffer.len() as f64;
                println!("this_compression_ratio: {:?}", this_compression_ratio);
                let mut decoder = FsstDecoder::new();
                let mut decompress_output_buffer: Vec<u8> = vec![0; 3 * compress_output_buffer.len() + 8096]; 
                //let mut decompress_output_buffer: Vec<u8> = vec![0; 32 * 1024 * 1024 + 8096]; 
                let mut decompress_offset_buffer: Vec<i32> = vec![0; 32 * 1024 * 1024 + 8096];
                decoder.decompress(&compress_output_buffer, &compress_offset_buffer, &mut decompress_output_buffer, &mut decompress_offset_buffer).unwrap();
                println!("decompress_output_buffer.len(): {}", decompress_output_buffer.len());
                println!("input.values().len(): {}", input.value_data().len());
                assert!(decompress_offset_buffer.len() == input.value_offsets().to_vec().len());
                assert!(decompress_output_buffer.len() == input.value_data().len());
                for i in 1..decompress_offset_buffer.len() {
                    let s = &decompress_output_buffer[decompress_offset_buffer[i-1] as usize..decompress_offset_buffer[i] as usize];
                    let original = &input.value_data()[input.value_offsets().to_vec()[i-1] as usize..input.value_offsets().to_vec()[i] as usize];
                    //println!("s: {:?}", std::str::from_utf8(s));
                    //println!("original: {:?}", std::str::from_utf8(original));
                    assert!(s == original, "s: {:?}\n\n, original: {:?}", std::str::from_utf8(s), std::str::from_utf8(original));
                }
            }
        }*/
    }

    use std::fs::File;
    use std::io::{BufRead, BufReader};

    fn read_random_16_m_chunk(file_path: &str) -> Result<StringArray, std::io::Error> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
    
        let lines: Vec<String> = reader.lines().collect::<std::result::Result<_, _>>()?;
        let num_lines = lines.len();
    
        let mut rng = rand::thread_rng();
        let mut curr_line = rng.gen_range(0..num_lines);
        let mut curr_line = 0;
        println!("curr_line: {}", curr_line);
    
        let chunk_size = 16 * 1024 * 1024; // 16MB
        let mut size = 0;
        let mut result_lines = vec![];
        while size + lines[curr_line].len() < chunk_size {
            result_lines.push(lines[curr_line].clone());
            size += lines[curr_line].len();
            curr_line += 1;
            curr_line %= num_lines;
        }
    
        Ok(StringArray::from(result_lines))
    }
    fn read_random_32_m_chunk(file_path: &str) -> Result<StringArray, std::io::Error> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
    
        let lines: Vec<String> = reader.lines().collect::<std::result::Result<_, _>>()?;
        let num_lines = lines.len();
    
        let mut rng = rand::thread_rng();
        let curr_line = rng.gen_range(0..num_lines);
        println!("curr_line: {}", curr_line);
        let mut curr_line = 2489500;
    
        let chunk_size = 32 * 1024 * 1024; // 32MB
        let mut size = 0;
        let mut result_lines = vec![];
        while size + lines[curr_line].len() < chunk_size {
            result_lines.push(lines[curr_line].clone());
            size += lines[curr_line].len();
            curr_line += 1;
            curr_line %= num_lines;
        }
    
        Ok(StringArray::from(result_lines))
    }
}