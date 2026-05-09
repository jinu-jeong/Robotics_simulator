# Paper positioning

**Decision:** target **ICRA 2027** (deadline ≈ mid-Sep 2026), single primary submission. Framing A folded into D. Systems-level contribution, not methods-level. arXiv preprint + GitHub release + Zenodo DOI in parallel; JOSS companion submission optional.

---

## The contribution sentence

> *We release **RoboSim**, an MIT-licensed pure-Python robotics simulator that integrates Featherstone articulated rigid-body dynamics, an MLS-MPM continuum solver (Neo-Hookean, von-Mises J2, Drucker–Prager, damaged-elastic), and Floating-Frame-of-Reference Craig–Bampton reduced flexible bodies under a single Scene API. The combination — classical CAE flex-body reduction sitting next to a Featherstone articulated robot and an MPM granular/plastic continuum, exposed through one Python API — is not present in MuJoCo / MJX, NVIDIA Isaac Sim/Lab, Drake, PyBullet, Brax, Dojo, Genesis, SOFA, ChainQueen / DiffTaichi / PlasticineLab, FluidLab, DiSECt, or SoftGym. We demonstrate the integration on a robot grasp-and-drop task in which a Craig–Bampton-reduced compliant box is grasped by a URDF arm and released onto an MPM clay or sand pile.*

This is a **systems / integration** claim, not a methods claim. Every algorithm is published prior art; the contribution is the packaging. ICRA's software-and-tools session evaluates exactly this kind of contribution.

---

## What is — and is not — original

| Element | Origin | Original to RoboSim? |
|---|---|---|
| Featherstone ABA / RNEA / CRBA | Featherstone (1987, 2008) | No |
| Corotational / Neo-Hookean FEM | Müller 2002, Bonet & Wood 2008 | No |
| MLS-MPM transfer kernel | Hu et al. SIGGRAPH 2018 | No |
| APIC affine velocity transfer | Jiang et al. SIGGRAPH 2015 | No |
| Drucker–Prager MPM, damaged-elastic MPM | Klár 2016, Wolper 2019 | No |
| Fixed-interface CMS (Craig–Bampton) | Craig & Bampton 1968 | No |
| FFR formulation around CMS | Likins 1967, Yoo & Haug 1986, Shabana 1997, Wallrapp 1994 | No |
| Per-step Kabsch rotation around linear elastic core | Müller et al. SCA 2002, James & Pai SIGGRAPH 2002 | No |
| Surface-keep boundary partition (full skin as master DOFs) | Engineering default in contact-rich CMS workflows | No (design choice) |
| **All of the above behind one Scene API alongside URDF and MPM** | — | **Yes — the integration** |

The "first integration in an open-source robotics-learning simulator" claim hinges on the qualifier. Without it, MSC Adams, Simpack, MotionSolve, Recurdyn, and Project Chrono have done equivalent integrations at the CAE-multibody level for ~30 years. With it, the audit in `cb_novelty.md` shows the niche is real.

**Do not claim:** "novel CMS variant," "novel FFR formulation," "first corotational reduced model," "first flex body in robotics," "first CB implementation," "novel reduction theory." All false.

---

## Three categories of prior art that must be acknowledged

The related-work section needs three explicit paragraphs so reviewers from each community see their work cited.

1. **CAE multibody simulators with FFR+CMS** — MSC Adams Flex, Simpack, MotionSolve, Recurdyn, Project Chrono (`chrono::modal`). These are the engineering ancestors. Cite Wallrapp 1994 (SID format), Shabana 2005 (textbook), Tasora et al. 2016 (Chrono).

2. **Flexible-link manipulator dynamics** — Book 1984 (recursive Lagrangian for flexible arms), De Luca & Siciliano 1991 (closed-form planar flex-link), Theodore & Ghosal 1995 (assumed-modes vs FE comparison), Wasfy & Noor 2003 (review), Bremer 2008 (textbook). Robotics has used FFR+modal-coords for flexible manipulator dynamics for 40 years, but as analytical models and bespoke MATLAB codes, not as a packaged simulator with URDF, contact, or learning-friendly Python.

3. **Open-source robotics-learning simulators** — MuJoCo / MJX, Isaac Sim/Lab, Drake, PyBullet, Brax, Dojo, Genesis, SOFA (+ SOFA-MOR plugin which is POD-hyperreduction, *not* CMS), ChainQueen, DiffTaichi, PlasticineLab, FluidLab, DiSECt, SoftGym. None of these expose CMS flex bodies. This is the gap RoboSim fills.

The contribution sentence is only defensible after all three paragraphs land. Drop any of them and a knowledgeable reviewer will reject on prior art.

---

## Target venue

**Primary: ICRA 2027** (software-and-tools session).

| Property | Value |
|---|---|
| Page limit | 6–8 (excluding references) |
| Deadline | ~mid-September 2026 (historical pattern: 14–20 Sep) |
| Review style | Single-blind, single-cycle (no major-revision), accept/reject |
| Time budget from now | ~4 months |
| Author identity | Visible — arXiv preprint with author names is fine |

**Why ICRA over RA-L:** wider robotics-community visibility, software-and-tools session is an established venue for simulator releases (PyBullet, Drake demos, MJX adjacent work all landed there), and the user prefers ICRA. The trade-off is the single annual deadline and no revision cycle.

**Backup if rejected:** RA-L tools track (rolling submission, allows major revision). Resubmit roughly January 2027 once ICRA decisions return. Do **not** resubmit to T-RO or IJRR — those venues require methods novelty and this paper does not have it.

**Companion artefacts (no venue conflict with ICRA):**
- arXiv preprint at submission time, citing GitHub `v0.1.0` tag and Zenodo DOI.
- JOSS submission against the GitHub repo — peer reviewed via GitHub issues, gives a separate citable software DOI. Low overhead, complements the ICRA paper.

**Do not target:** Multibody System Dynamics (Springer), ASME J. Mechanisms & Robotics — wrong audience; reviewers there will demand comparison against Adams Flex / Simpack / Chrono and the framing collapses.

---

## Submission timeline

| Month (rough) | Milestone |
|---|---|
| 2026-05 (now) | Branch frozen, README + survey done, paper folder seeded |
| 2026-05 → 2026-06 | CB validation experiments (vs Chrono or analytical Euler–Bernoulli beam) |
| 2026-06 → 2026-07 | FPS benchmark vs MuJoCo / MJX / PlasticineLab / Genesis on standard scenes |
| 2026-07 → 2026-08 | Three-solver demo: URDF arm + CB-reduced compliant link + MPM clay block |
| 2026-08 | First full draft, internal review |
| 2026-08 → 2026-09 | Revision, Zenodo DOI, arXiv submission |
| 2026-09 | ICRA 2027 submission |
| 2026-09 → 2026-10 | JOSS submission (parallel) |
| 2027-01 | ICRA decisions; if rejected, prepare RA-L resubmission |

---

## Required additional work

To turn the current branch into the paper artefact:

0. **Fix the pure-simulation examples (prerequisite for everything below).** `examples/pure_simulation/` (drop_test, mpm_grasp_drop, fem_cantilever_taichi, mpm_jello_drop, mpm_plasticity_demo, mpm_clay_tear, hybrid_test, interactive_demo, load_urdf_demo, grasp_scene) must each: run end-to-end on a clean conda env, produce the documented output (banner, success print, no exceptions), and have one short comment header explaining what to look for. These are the building blocks RL builds on, the headline demos for the paper, *and* the smoke tests for any future refactor — if they don't run cleanly, the RL story collapses too. Treat as the first deliverable.
1. **CB validation experiment.** One figure showing RoboSim's FFR+CMS body matches an analytical reference (Euler–Bernoulli cantilever first 3 mode shapes and tip deflection under gravity), and one figure comparing tip-deflection trajectory against Project Chrono `chrono::modal` for a more complex geometry.
2. **FPS benchmark table.** Columns: simulator. Rows: rigid drop, FEM cantilever (where supported), MPM jelly, MPM sand, full grasp-drop. Compare RoboSim against MuJoCo / MJX / PyBullet / PlasticineLab / Genesis on identical scene parameters. Be honest — RoboSim will be slower than Genesis and MJX. The pitch is reproducibility and modifiability, not raw speed.
3. **Three-solver headline demo.** URDF arm grasps a CB-reduced compliant box and drops it on an MPM clay/sand pile. A scene none of the other listed simulators can express end-to-end. Video figure for the paper, GIF for the README.
4. **Honest evidence table** (already drafted in [cb_novelty.md](cb_novelty.md) §2) — converted into a paper-quality LaTeX table.
5. **Related-work section** with the three paragraphs above, citing all of `references.bib`.
6. **Software hygiene:** CI badge, code coverage, Zenodo DOI button, MIT license file, contribution guide, semver policy.
7. **One short tutorial appendix** ("30-line MPM in pure Python") demonstrating the readability claim.

Estimated total effort: ~3–4 months working alongside other research, comfortable with the September 2026 ICRA deadline.

---

## Risk register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Reviewer says "Genesis already does all this, faster" | High | Medium | Pre-empt in intro: *"Genesis (2024) targets production performance via Taichi/CUDA; we target reproducibility and modifiability via pure Python."* Cite specifically and respectfully. |
| Reviewer says "Adams Flex / Chrono have had this for decades" | High | Medium | Lead with the qualifier "in open-source robotics-learning simulators" in the abstract; cite Wallrapp 1994, Tasora 2016, Shabana 2005 prominently. |
| Reviewer says "Book 1984 + assumed-modes is the same thing" | Medium | High | Acknowledge in related-work paragraph 2; emphasise simulator packaging vs analytical model + URDF + contact + MPM as the integrated artefact. |
| Reviewer says "no methods novelty" | High | High | Submit to ICRA software-and-tools or RA-L tools — venues where systems-level integration is the accepted contribution type. Do not submit to T-RO/IJRR. |
| MJX or Genesis ships CB before submission | Low | High | Watch their release notes monthly. If it happens, pivot to Framing B (one-way coupling study) or Framing C (GNN-ROM). |
| FPS comparison embarrasses RoboSim | Medium | Low | Lead with the trade-off explicitly: pure-Python is a feature, not an oversight. |

---

## Out-of-scope framings (kept for context)

The earlier discussion considered three other framings; none are the current target.

- **Framing B: one-way coupling study.** A characterisation paper on when one-way RBD→MPM coupling is sufficient for manipulation. Future IROS submission, citing this paper as the simulator.
- **Framing C: GNN-ROM in RoboSim.** Replaces classical CB with a learned graph-neural ROM. Future CoRL/RSS submission, citing this paper as the CB baseline. Gated on the `feat/gnn-boundary-rom` research outcomes.
- **Framing originally labelled A: tools paper without the CB hook.** Subsumed into the current submission — the CB integration *is* the headline because it gives the paper a narrow but specific niche claim instead of a generic "we made a simulator."
