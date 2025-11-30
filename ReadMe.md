# Waterbird

> You know those videos of herons eating snakes? 

Waterbird is another toy language of mine, intended for maximum useability for me to solve algorithmic problems quickly (coding speed, not execution speed).


# Use it
##### I highly advise against using this language without accepting its many undocumented quirks

Compile the compiler
```bash
cargo build --release
```

Run the program "test_sqrt" in the folder "programs"
```bash
./target/release/waterbird-lang programs/test_sqrt
```
Note: file-reads are relative to the project root


# Features

## Interpreter
- Manual-memory collection (deallocate with ``dealloc(object)``)
- dynamic typing
- wacky collection of undocumented built-in functions
- rust-style closures with automatic capturing of context
- hashable tuples


### Disclaimer
I use LLMs in the process of developing this, both as rubber duckies, automation of boilerplate but also to write core logic. 
Estimate: ~10% of this repositories code stems directly from Gemini (and maybe ~1% ChatGPT), although basically everything was adjusted over time.