# Lamb

Tiny Pure Functional Programming Language in Rust. Based on [Untype Lambda Calculus](https://en.wikipedia.org/wiki/Lambda_calculus) with Normal Order reduction.

This project is based on an [original work](https://github.com/tsoding/lamb/) by **Alexey Kutepov** which is written in C. Refer to its [README](https://github.com/tsoding/lamb/blob/main/README.md) for full documentation.

## Quick Start

```bash
$ rustc -o lamb lamb.c
$ ./lamb ./std.lamb
...
,---@>
 W-W'
 Enter :help for more info
 @> pair 69 (pair 420 1337)
 RESULT: \f.f 69 (\f.f 420 1337)
 @> xs = pair 69 (pair 420 1337)
 Created binding xs
 @> first xs
 RESULT: 69
 @> second xs
 RESULT: \f.f 420 1337
 @> first (second xs)
 RESULT: 420
 @>
 ```

## Main difference

This Rust implementation does not feature a garbage collector, but instead takes advantage of Rust's ownership system for memory management. In particular, it wraps expression objects in an `Rc`, which is a reference-counted pointer. The object is freed automatically when all references go out of scope. See the [Rust documentation](https://doc.rust-lang.org/stable/std/rc/struct.Rc.html) for more information.

## Shortfalls

- This version does not currently handle "SIGINT" signals on unix systems. Since Rust doesn't yet have built-in abstractions for signal handling, I'm torn between using a recommended library (introducing excess code I don't fully understand), or writing one myself (incredibly onerous, but educational). At this point, I'm keeping it simple. `Ctrl-C` just quits the program.

- This hasn't yet been tested on Windows.

## Why I made this

Practice, self-education, and skill improvement.

## Feedback

If you try this out and find an error or inconsistency, feel free to [open an issue](https://github.com/emamoah/lamb/issues/new).
