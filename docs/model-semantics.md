# Graphion Model Semantics

**Status: Approved.**

This document defines the shared modeling rules.
The [Programming Reference](reference/index.md) describes the implemented API.
**MUST** and **MUST NOT** express requirements; **MAY** permits a choice.
Capitalized terms have the definitions below, including their plural forms.

## Author guidance

| Need | Do |
| --- | --- |
| A pure function of current Inputs | Use Output Evaluation without State Update ([S7](#s7-split-behavior)). |
| History across Block Iterations | Use State with an Initial Value ([S3](#s3-state)). |
| Clear history from a signal | Use a declared Input and an ordinary State transition ([S9](#s9-reset)). |
| Outputs and history from one calculation | Use Atomic Form ([S6](#s6-determinism-and-atomic-behavior)). |
| Break a feedback cycle | Use Split Form whose Output Evaluation needs no Input from the cycle at the chosen granularity ([S10](#s10-feedback-and-granularity)). |
| A setting that changes while running | Use a Control, changed between Block Iterations ([S2](#s2-configuration)). |
| Several Blocks as one | Use a Composite with recorded Connections and coordinated execution ([S4](#s4-hierarchy)). |

## Defined terms

| Term | Definition |
| --- | --- |
| Model | Blocks, Connections, configuration, and behavior describing a processing task. |
| Block | A processing instance with interfaces, configuration, and optional State. |
| Composite | A Block that owns and coordinates other Blocks. |
| Child | A Block registered under an owning Composite. |
| Parameter | Configuration fixed for a configured Block. |
| Control | Configuration that may change between Block Iterations. |
| State | Block-owned retained data whose previous value affects subsequent behavior, excluding Inputs and configuration. |
| Initial Value | The value established for State by initialization and restored by Reset. |
| Scratch | Workspace that supplies no history across Block Iterations. |
| Port | A named input or output interface declaration. |
| Input | A value received through a Port. |
| Output | A result provided through a Port. |
| Connection | An explicit record identifying a source Port and a destination Port. |
| Input Dependency | An Input required before a particular Phase can run. |
| Execution Model | Rules coordinating execution and communication, realized by ordinary caller code or automation. |
| Schedule | The order and cadence of processing invocations. |
| Block Iteration | One logical unit of a Block's processing and associated State advancement, if any. |
| Phase | Atomic processing, Output Evaluation, or a separate State Update. |
| Atomic Form | One operation that produces Outputs and may advance State. |
| Split Form | Output Evaluation followed by an optional separate State Update. |
| Output Evaluation | Computation of Outputs for a Block Iteration. |
| State Update | Establishing State for subsequent Block Iterations. |
| Direct Producer | The Block responsible for computing an Output. |
| Borrow | Access to storage for a stated lifetime without ownership transfer. |
| Modeled Effect | A modeled action and associated values, such as changing State or Controls, requesting external I/O, or advancing a random generator. |
| Reset | Abandoning pending work and restoring State to its Initial Values within a stated scope. |
| Implementation Profile | A specialization of the shared Block model defining concrete representations, execution, ownership, storage, lifetime, and lifecycle contracts. |

## Execution policy

A Model need not declare, name, or instantiate an Execution Model or
Implementation Profile. Implementation Profiles and Execution Models specify
binding, timing, Input values associated with a Block Iteration, completion,
Control coordination, Reset scope, and failure recovery, subject to the shared
rules below.

**User Defined Execution Model** describes ordinary caller-written execution.
The caller chooses the Schedule, including conditional calls and Phase
interleaving, and MUST satisfy the applicable contracts. A violating Schedule
does not establish valid modeled behavior; specified validation and recovery
behavior still applies. There is no implicit whole-Model iteration, fresh-value
consumption, global commit, automatic scheduling, or proof of Schedule correctness.

## Shared rules

### S1 Ports and connections

Ports MUST NOT supply algorithmic history. Connections MUST identify actual
Input sources, and binding MUST preserve those sources. Rebinding MUST reach
affected invocations. Neither a Connection nor bound storage establishes
freshness, a Schedule, or a delay.

### S2 Configuration

Parameters MUST remain fixed for a configured Block. Controls, including mutable
contents, MUST remain stable throughout its Block Iteration.

### S3 State

State MUST have defined Initial Values before processing reads it; defaults or
values derived from configuration suffice. Only the owning Block MAY initialize
or modify its State. A Block's processing MUST NOT directly read another Block's
State. Counters and random-generator history are State regardless of storage or
declaration syntax. Scratch MUST NOT supply history across Block Iterations.
Except for Reset or specified failure recovery, each Block Iteration MUST start
with the State left by its predecessor.

### S4 Hierarchy

A Child has one lifecycle owner. A Composite coordinates Child operations;
hierarchy alone MUST NOT authorize replacing, flattening, or reordering opaque
processing code. Transformations require explicit authorization and MUST preserve
the contracted behavior. Boundary forwarding MUST preserve the Direct Producer
and Output lifetime and MUST NOT add history.

### S5 Input dependencies

The Implementation Profile MUST specify how Input Dependencies are declared.
Each Phase MUST satisfy its declared dependencies before executing and MUST NOT
read undeclared live Inputs. A Composite's declarations MUST include its Inputs
consumed through Child operations. Each Phase's readiness MUST use its declared
Input Dependencies. An Input required by State Update but excluded from Output Evaluation's dependencies
MUST NOT delay Output Evaluation.

### S6 Determinism and atomic behavior

For identical declared Input values, configuration, starting State, and Schedule,
a Block MUST produce the same Outputs, resulting State, and ordered sequence of
Modeled Effects under its Implementation Profile's numeric contract. External
outcomes are not guaranteed; observations affecting behavior MUST enter through
declared Inputs.
Each successful output-producing evaluation MUST fully define all directly
produced Outputs. Omission MUST NOT imply retention of previous results.

In Atomic Form, one completed Block Iteration consists of one successful
invocation, which MAY advance State. No Output from that invocation is available
before its required Inputs. Atomic Form does not imply transactional rollback.

### S7 Split behavior

A Block chooses one authoring form. A completed Split Form Block Iteration
consists of one successful Output Evaluation followed by exactly one State Update,
when defined. Without State Update, successful Output Evaluation completes the
iteration and State is unchanged. Output Evaluation MAY write result storage and
Scratch but MUST NOT produce other Modeled Effects, including advancing Child
State or changing Controls, even if it fails. Purity is an author contract.

State Update MUST use the values observed by Output Evaluation for Inputs required
by both Phases, with the same configuration. This concerns values, not storage
identity. It MUST NOT produce or modify Outputs. Failure does not complete the
iteration ([S11](#s11-failure)).

### S8 Output stability and update ordering

The applicable contract MUST define when an Output becomes available and when
its Borrow expires. Outputs MUST remain stable during that Borrow; State Update
MUST NOT invalidate current Outputs. Consumers MUST finish reads before storage
is reused, including reads during State Updates. A Borrow is not a snapshot.
State Updates MAY be reordered only when Inputs remain valid and Modeled Effects
permit the order.

### S9 Reset

Reset MUST first abandon all pending Block Iterations within its stated scope,
then restore Initial Values. Subsequent processing begins a new Block Iteration.
Reset does not implicitly change Parameters, Controls, or Connections, complete
abandoned work, or undo external effects already performed.
Implementation Profiles define Borrow expiration.

Processing MUST NOT invoke Reset. During normal processing, a Block MAY restore
selected State to Initial Values as an ordinary State transition. This does not
constitute Reset.

### S10 Feedback and granularity

State breaks a feedback dependency only when the needed Output can be provided
before the corresponding Input is available. This depends on processing
granularity: a one-sample delay can still require current frame data.

### S11 Failure

A failed invocation MUST NOT be reported as a completed Block Iteration.
Partial writes and Modeled Effects may remain. The Implementation Profile MUST
define failure status and the conditions permitting subsequent execution.

### S12 Descriptions and claims

Descriptions MUST use authoritative Connections, distinguish known facts from
opaque behavior, and MUST NOT advance State. Conformance claims MUST identify
their scope; declarations and selected tests MUST NOT be presented as proof of
arbitrary author code or complete dependency information.

## NumPy contract

The NumPy frame Implementation Profile specifies [execution and failure](reference/execution.md),
[Reset and lifecycle](reference/lifecycle.md), and
[storage and Borrow lifetime](reference/storage-and-lifetime.md#the-buffer-contract).
These concrete rules specialize the shared semantics above.

## Acceptance examples

These cases assume caller-ordered synchronous execution and the NumPy
contract above. Values are exact; a number denotes a constant frame except in A6.
A whole-frame delay starts with the stated Initial Value, evaluates its held
State, and then captures its Input. Its whole Block Iteration includes both Phases.

| Case | Initial Values, Inputs, calls, and expected results |
| --- | --- |
| A1 Delay | Initial Value `0`, Input unassigned: Output Evaluation gives `0`. Bind `5`, then State Update: Output stays `0`. Next Output Evaluation gives `5`. |
| A2 Series | Two delays start at `0`. Supply upstream Inputs `1, 2, 3`; for each, run one whole upstream Block Iteration then one downstream Block Iteration. Downstream Outputs are `0, 0, 1`. |
| A3 Feedback | Delay starts at `0`; adder sums the external Input and half the delay Output. For Inputs `1, 0, 0, 0`, evaluate delay, run adder, then update delay from adder. Adder Outputs are `1, 0.5, 0.25, 0.125`; Reset and replay give the same results. |
| A4 Exchange | Delays start at `1` and `2`, each receiving the other's Output. Evaluate both, then update both in either order: current Outputs stay `1, 2`. Next Output Evaluations give `2, 1`. |
| A5 Wrapped delay | A Composite uses Split Form, forwards its Input to a delay initialized to `0`, and exposes the delay's Output. With boundary Input unassigned, its Output Evaluation calls the Child's and gives `0`. Bind `7` and invoke its State Update, which calls the Child's; next Output Evaluation gives `7`. |
| A6 Granularity | A one-sample delay starts at `0`. Whole invocations with frames `[1, 2, 3]` then `[4, 5, 6]` produce `[0, 1, 2]` then `[3, 4, 5]`. Each Output needs current frame data, so this delay cannot break a whole-frame feedback dependency. |
| A7 Shared Input | An accumulator starts at `0`. Output Evaluation computes State plus Input; State Update adds Input to State. With Input `2`, Output Evaluation gives `2` and leaves State at `0`; update with `2` sets State to `2` and leaves Output unchanged. In the next iteration, Input `5` gives Output `7`, then State `7` after update. |
