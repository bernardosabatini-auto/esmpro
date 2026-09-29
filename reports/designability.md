# designability — are the structures we generate real proteins?

**Loop** structure -> ProteinMPNN (CA-only model, 8 sequences per backbone, T = 0.1) -> ESMFold2-Fast -> TM back to the structure we started from. Best over the 8 sequences, designable if above 0.5, the RFdiffusion / Genie protocol. ProteinMPNN's CA-only weights take exactly what our decoder produces, so no backbone is fabricated. **Coverage 1.00 on every row.** Jobs 49156040-51 (`code/gate31_designability_mpnn.py`), 64 or 32 backbones per set.

| structures | model | n | scTM | median | designable |
|---|---|---|---|---|---|
| **real experimental (control)** | — | 64 | **0.854** | 0.916 | **94 %** |
| unconditional prior | 459M | 32 | **0.609** | 0.627 | **75 %** |
| noise interpolation, midpoint | 459M | 32 | **0.629** | 0.649 | **69 %** |
| unconditional prior | 840M | 64 | 0.548 | 0.543 | 56 % |
| contact-steered (8 A constraint) | 459M | 32 | 0.494 | 0.488 | 44 % |
| motif inpainting (30 % held) | 459M | 32 | 0.398 | 0.385 | 6 % |

## Reading
1. **The pipeline is sound and the earlier attempt was not.** Real structures score 0.854 with 94 % designable; through our own 174M co-design head the same real structures scored 0.376. `codesign.md`'s inverse-folding numbers measure that head, not the structures, and the gate29 comparison should be read as void.
2. **Unconditional samples are genuinely designable.** Given only a length and the null token, the 459M model produces structures that ProteinMPNN can write a sequence for and ESMFold2 folds back to within TM 0.61 on average, 75 % of them above 0.5. This is the first result in the project that is a *generation* claim rather than a folding claim, and it is on an independent designer and an independent folder.
3. **Noise-space interpolation keeps designability** (0.629 / 69 %, indistinguishable from the prior's 0.609 / 75 % at these sample sizes). The path between two samples is not just geometrically valid, it is designable along its length. Interpolating finished latents, by contrast, was already geometrically dead (`steering.md`).
4. **Steering costs designability even when geometry is perfect.** The contact-steered set satisfies its 8 A constraint with Ca-Ca 3.84 and zero clashes, yet designability falls from 75 % to 44 %. Bond geometry and clash counts do not detect this; only the design loop does. Any future steering knob must be scored this way, not on geometry.
5. **Inpainting fails.** Holding 30 % of residues on a real motif's path gives 6 % designable, the worst of every set including perturbed latents. The scaffold around a fixed motif is not forming a coherent protein. Since motif scaffolding is the most useful design capability, this is the clearest target for work.
6. The 459M prior beats the 840M prior (0.609 / 75 % against 0.548 / 56 %), consistent with `scaling_exhausted.md`: the larger model is not better here either.

## What it changes
The generative claim now has an independent number: 75 % of unconditional samples from a 459M latent flow are designable, against 94 % for real structures through the same pipeline. Steering satisfies constraints but costs designability, and motif scaffolding does not yet work. Those two, not folding accuracy, are where the generative line should go next.
