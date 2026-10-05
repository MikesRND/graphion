# Graphion Charter

This charter says why the project exists and which principles every
design must keep. Contracts and design decisions live in the
[programming reference](reference/index.md), and implementation status in the
[README](../README.md).

Amend this charter narrowly, only when necessary to address concrete
needs, correct inaccuracies, or agreed scope changes.

## Purpose

Graphion aims to be a layer in which a person or an AI agent can express a
model that a person can read, visualize, understand and review. Model
execution can be user-defined, as today, or automated under a chosen model of
computation. Automated execution lets a tool validate a model before it runs
and generate code that is correct with respect to that model of computation.
This allows human review to happen at the level of the model rather than the
generated code.

Models are composed from blocks with declarative interfaces, explicit state
semantics, predictable storage use and visible structure. The block contracts
should not depend on the implementation technology: Python, C++, CUDA, HDL or
others.

## Principles

1. **Authoring stays simple.** A block is a few declarations and its
   processing operation. A simple gain block must stay simple.

2. **Execution is explicit.** Under user-defined execution, whoever writes
   or owns a model decides what runs and when. Automated execution follows
   the model of computation its author chose. Nothing is scheduled, reordered
   or repeated silently.
   Showing a model's structure does not give another tool permission to
   change how it runs.

3. **Storage is predictable.** Storage needs and reuse are explicit.
   When interfaces and lifetimes permit sharing, the framework imposes no
   mandatory per-step data-buffer allocation or composite-boundary copy.
   Required transfers, conversions and retention are explicit, and
   efficient code uses the same interface as convenient code.

4. **Data ownership and lifetime are stated.** It is always clear who may
   write a buffer, how long a result stays valid and how data may be
   retained.

5. **Structure is visible.** A model's blocks, ports, connections and
   hierarchy can be inspected and drawn without executing a processing
   step, and described in a form other tools can use. Connections are
   explicit records, never inferred from how memory happens to be shared.
   A description says what it does not know.

6. **The model does not depend on one implementation.** The same concepts
   should be realizable in different technologies. Each implementation
   states its own numeric, timing and reset behavior. Implementations are
   checked against shared examples, not assumed to be interchangeable.

7. **Cost is measured.** Framework overhead is kept small and shown with
   recorded evidence, separately from the work performed by blocks.

8. **The core stays small.** It should fit inside larger systems and
   depend on none of them. An extension is added when a concrete need
   exists, and it must not burden ordinary blocks.

## Scope

The project serves general processing models, with RF/DSP as one
application. It provides a block model and implementations of it.
Schedulers and model compilers can be built around it and can consume
its structural descriptions.

Automatic scheduling, inferred timing or causality, memory planning and
automatic translation between implementations are outside the core.

## Review question

A proposal should name the concrete use it serves, its effect on
ordinary block authoring, its storage and execution consequences, and how
it will be verified.

> Does this make a real processing model easier to write, understand, inspect,
> check or run efficiently, while keeping a simple Gain block simple and leaving
> its implementation technology a deliberate choice?
