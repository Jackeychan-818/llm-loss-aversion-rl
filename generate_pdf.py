"""
Generate training_overview.pdf using matplotlib (for math) + fpdf2 (for layout).
Renders each section as a matplotlib figure, saves to PDF pages.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from fpdf import FPDF          # package is "fpdf2" on PyPI, module is "fpdf"
import tempfile, os

# ── colour palette ──────────────────────────────────────────────────────────
BG      = '#FFFFFF'
TEXT    = '#1a1a1a'
ACCENT  = '#2c5f8a'
BOX_BG  = '#f0f4f8'
BOX_BD  = '#90b4d4'
GREEN   = '#2a7a2a'

def fig_to_tmp(fig, dpi=180):
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    fig.savefig(tmp.name, dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    return tmp.name

def make_section_fig(title, content_fn, figsize=(8.27, 11.69)):
    fig = plt.figure(figsize=figsize, facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    content_fn(ax)
    return fig

# ── helpers ─────────────────────────────────────────────────────────────────
def title_text(ax, y, text, size=18, color=ACCENT, bold=True):
    weight = 'bold' if bold else 'normal'
    ax.text(0.05, y, text, transform=ax.transAxes,
            fontsize=size, color=color, fontweight=weight,
            verticalalignment='top')

def body(ax, y, text, size=9.5, color=TEXT, x=0.05, wrap=95):
    import textwrap
    lines = []
    for para in text.split('\n'):
        if para.strip() == '':
            lines.append('')
        else:
            lines.extend(textwrap.wrap(para, wrap))
    full = '\n'.join(lines)
    ax.text(x, y, full, transform=ax.transAxes,
            fontsize=size, color=color, verticalalignment='top',
            fontfamily='monospace' if '    ' in text else 'sans-serif',
            linespacing=1.55)

def math(ax, y, expr, size=11, x=0.5):
    ax.text(x, y, expr, transform=ax.transAxes,
            fontsize=size, color=TEXT, verticalalignment='top',
            ha='center', math_fontfamily='dejavuserif')

def hline(ax, y, lw=0.8, color=ACCENT, xmin=0.05, xmax=0.95):
    # axhline rejects an explicit transform on current matplotlib; the axes are
    # set to xlim/ylim (0, 1) with axis off, so axes coords == data coords here.
    ax.axhline(y, xmin=xmin, xmax=xmax, color=color, linewidth=lw)

def box(ax, x0, y0, w, h, text, size=8.5):
    rect = FancyBboxPatch((x0, y0-h), w, h,
                          boxstyle='round,pad=0.01',
                          facecolor=BOX_BG, edgecolor=BOX_BD, linewidth=1,
                          transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(x0 + w/2, y0 - h/2, text, transform=ax.transAxes,
            fontsize=size, color=TEXT, ha='center', va='center',
            linespacing=1.5)

def table(ax, y, headers, rows, col_widths, x0=0.05, row_h=0.042, size=8.5):
    # header row
    xpos = x0
    for h, w in zip(headers, col_widths):
        ax.add_patch(FancyBboxPatch((xpos, y-row_h), w-0.005, row_h,
                                    boxstyle='square,pad=0',
                                    facecolor=ACCENT, edgecolor='none',
                                    transform=ax.transAxes))
        ax.text(xpos + (w-0.005)/2, y - row_h/2, h,
                transform=ax.transAxes, fontsize=size, color='white',
                ha='center', va='center', fontweight='bold')
        xpos += w
    y -= row_h
    for ri, row in enumerate(rows):
        xpos = x0
        bg = '#e8f0f8' if ri % 2 == 0 else BG
        for val, w in zip(row, col_widths):
            ax.add_patch(FancyBboxPatch((xpos, y-row_h), w-0.005, row_h,
                                        boxstyle='square,pad=0',
                                        facecolor=bg, edgecolor='#cccccc',
                                        linewidth=0.3, transform=ax.transAxes))
            ax.text(xpos + (w-0.005)/2, y - row_h/2, str(val),
                    transform=ax.transAxes, fontsize=size, color=TEXT,
                    ha='center', va='center')
            xpos += w
        y -= row_h
    return y

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Title + Overview + Experimental Paradigm
# ═══════════════════════════════════════════════════════════════════════════
def page1(ax):
    # title block
    ax.add_patch(FancyBboxPatch((0.03, 0.85), 0.94, 0.13,
                                boxstyle='round,pad=0.01',
                                facecolor=ACCENT, edgecolor='none',
                                transform=ax.transAxes))
    ax.text(0.5, 0.935, 'Training Qwen2.5-7B with GRPO to Reduce Loss Aversion',
            transform=ax.transAxes, fontsize=16, color='white',
            fontweight='bold', ha='center', va='center')
    ax.text(0.5, 0.875, 'Training Process Overview — lambda-zero',
            transform=ax.transAxes, fontsize=11, color='#d0e4f4',
            ha='center', va='center')

    y = 0.82
    title_text(ax, y, '1  Overview', size=13); y -= 0.045
    hline(ax, y); y -= 0.018
    body(ax, y, (
        'This document describes the end-to-end pipeline used to fine-tune Qwen2.5-7B-Instruct\n'
        'via Group Relative Policy Optimization (GRPO) to make more rational trade decisions.\n\n'
        'The pipeline has four stages:\n'
        '  1.  Construct a dataset of trade scenarios with ground-truth utility differences δ̃\n'
        '  2.  Define a reward function that scores each model response against δ̃\n'
        '  3.  Run GRPO to update the model\'s LoRA parameters using those rewards\n'
        '  4.  Evaluate on a held-out test set and estimate λ̂_after'
    )); y -= 0.195

    title_text(ax, y, '2  Experimental Paradigm', size=13); y -= 0.042
    hline(ax, y); y -= 0.018
    body(ax, y, (
        'Each training example presents the model with an endowment-effect trade scenario.\n'
        'The model is endowed with good X (or Y) and asked whether it would trade for the other good.\n\n'
        'Each case runs from BOTH perspectives:\n'
        '  • X-perspective: endowed with X, offered Y\n'
        '  • Y-perspective: endowed with Y, offered X\n\n'
        'A rational agent\'s decision depends only on which good is genuinely better, not on\n'
        'which one it currently holds. The loss aversion coefficient λ measures how much the\n'
        'model\'s choices deviate from this rational benchmark.'
    )); y -= 0.215

    # example prompt box
    ax.add_patch(FancyBboxPatch((0.05, y-0.155), 0.90, 0.155,
                                boxstyle='round,pad=0.012',
                                facecolor='#fff8e8', edgecolor='#e0b840',
                                linewidth=1.2, transform=ax.transAxes))
    ax.text(0.07, y-0.012,
            'Example Prompt',
            transform=ax.transAxes, fontsize=8, color='#996600',
            fontweight='bold', va='top')
    ax.text(0.07, y-0.032,
            ('Suppose that, by chance, you receive a shoe polish out of many possible\n'
             'goods you could have received.  In terms of finish, it is shiny; in terms\n'
             'of formula, it is enriched.  You can keep the shoe polish or trade it for\n'
             'a highlighter.  In terms of saturation, the highlighter is bold; in terms\n'
             'of tip, it is firm durable.\n'
             'Would you trade it? Answer with the single word Yes or No.'),
            transform=ax.transAxes, fontsize=8.5, color='#4a3000',
            va='top', linespacing=1.6, style='italic')

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Dataset + Delta
# ═══════════════════════════════════════════════════════════════════════════
def page2(ax):
    y = 0.96
    title_text(ax, y, '3  Dataset', size=13); y -= 0.042
    hline(ax, y); y -= 0.02
    body(ax, y, (
        'The full dataset contains 59,400 paired trade scenarios, split as follows:'
    )); y -= 0.055

    # dataset table
    y = table(ax, y,
              ['File', 'Cases', 'Case IDs', 'Purpose'],
              [['trial_goods.json',     '60',     '1–60',        'Sanity checks'],
               ['test_goods.json',      '9,890',  '61–9,950',    'Held-out evaluation (λ̂_after)'],
               ['remaining_goods.json', '49,450', '9,951–59,400','GRPO training']],
              [0.32, 0.12, 0.18, 0.33], row_h=0.038)
    y -= 0.02
    body(ax, y, (
        'The model is trained only on remaining_goods.json. λ̂_after is estimated on the\n'
        'held-out test set — cases the model never saw during training.'
    )); y -= 0.075

    title_text(ax, y, '4  Ground-Truth Utility Differences (δ̃)', size=13); y -= 0.042
    hline(ax, y); y -= 0.02

    title_text(ax, y, '4.1  Why We Need δ̃', size=11, color=TEXT, bold=False); y -= 0.035
    body(ax, y, (
        'To reward rational behaviour, we need to know which good is actually better in each\n'
        'scenario. We operationalise this as the utility difference:\n'
    )); y -= 0.065
    math(ax, y, r'$\delta = U_X - U_Y$', size=12); y -= 0.05
    body(ax, y, (
        'A positive δ means X is genuinely better; negative means Y is better.\n\n'
        'We CANNOT estimate δ from Qwen-7B\'s own responses because the base model says\n'
        '"No" 99% of the time — its choices carry almost no information about which good is\n'
        'actually better, so its utility estimates are unreliable.'
    )); y -= 0.115

    title_text(ax, y, '4.2  Estimating δ̃ from Frontier Models', size=11, color=TEXT, bold=False)
    y -= 0.035
    body(ax, y, (
        'Instead, δ is estimated from six frontier models that actually trade at reasonable\n'
        'rates: GPT-4o, GPT-5, GPT-5.2, Gemini 1.5 Pro, Llama-70B, and DeepSeek-R1.\n'
        '(Claude, Apertus-70B, and GPT-3.5 are excluded — their <5% Yes-rates make their\n'
        'utility estimates unreliable for the same reason as Qwen-7B.)\n\n'
        'For each frontier model m, we run the structural NLS estimation (Model A) to obtain\n'
        'item- and attribute-level coefficients α̂⁽ᵐ⁾, β̂⁽ᵐ⁾.  The utility of good g is:'
    )); y -= 0.165
    math(ax, y, r'$\hat{U}^{(m)}_g = \hat{\alpha}^{(m)}_g + \sum_{k=1}^{2} \hat{\beta}^{(m)}_{k,\, a_{gk}}$',
         size=12); y -= 0.055
    body(ax, y, 'where a_{gk} is the attribute level of good g on dimension k.'); y -= 0.045
    body(ax, y, 'The consensus utility difference for case i is then:'); y -= 0.038
    math(ax, y, r'$\tilde{\delta}_i = \frac{1}{6} \sum_{m=1}^{6} \left(\hat{U}^{(m)}_{X_i} - \hat{U}^{(m)}_{Y_i}\right)$',
         size=12); y -= 0.06
    body(ax, y, (
        'All 59,400 values of δ̃ are precomputed and stored in\n'
        'data/deltas/delta_consensus_v3.json.  About 64% of cases have unanimous sign\n'
        'agreement across all six models; the remaining 36% use the mean δ̃, which naturally\n'
        'receives lower magnitude and thus lower gradient weight during training.'
    ))

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Reward Function
# ═══════════════════════════════════════════════════════════════════════════
def page3(ax):
    y = 0.96
    title_text(ax, y, '5  Reward Function', size=13); y -= 0.042
    hline(ax, y); y -= 0.022

    title_text(ax, y, '5.1  Rational Answer', size=11, color=TEXT, bold=False); y -= 0.038
    body(ax, y, 'Given δ̃ᵢ = Û_Xᵢ − Û_Yᵢ, the rational answer for each perspective is:'); y -= 0.045

    # rational answer box
    rules = (
        'δ̃ > 0  →  X is better\n'
        '   X-perspective:  rational = "No"   (keep X)\n'
        '   Y-perspective:  rational = "Yes"  (trade for X)\n\n'
        'δ̃ < 0  →  Y is better\n'
        '   X-perspective:  rational = "Yes"  (trade for Y)\n'
        '   Y-perspective:  rational = "No"   (keep Y)'
    )
    ax.add_patch(FancyBboxPatch((0.05, y-0.175), 0.90, 0.175,
                                boxstyle='round,pad=0.012',
                                facecolor=BOX_BG, edgecolor=BOX_BD,
                                linewidth=1, transform=ax.transAxes))
    ax.text(0.09, y-0.018, rules, transform=ax.transAxes,
            fontsize=9.5, color=TEXT, va='top',
            fontfamily='monospace', linespacing=1.7)
    y -= 0.195

    title_text(ax, y, '5.2  Reward Formula', size=11, color=TEXT, bold=False); y -= 0.038
    body(ax, y, 'The reward for a model response r̂ᵢ is:'); y -= 0.045

    # reward formula box (highlighted)
    ax.add_patch(FancyBboxPatch((0.12, y-0.115), 0.76, 0.115,
                                boxstyle='round,pad=0.015',
                                facecolor='#eaf4ea', edgecolor='#4a9a4a',
                                linewidth=1.5, transform=ax.transAxes))
    math(ax, y-0.03,
         r'$\mathcal{R}(\hat{r}_i,\,\tilde{\delta}_i) = \begin{cases} +|\tilde{\delta}_i| & \text{if } \hat{r}_i = r^*_i \text{ (rational)} \\ -|\tilde{\delta}_i| & \text{if } \hat{r}_i \neq r^*_i \text{ (irrational)} \end{cases}$',
         size=12); y -= 0.13

    title_text(ax, y, '5.3  Why Use |δ̃| as Magnitude?', size=11, color=TEXT, bold=False); y -= 0.038
    body(ax, y, (
        'Using |δ̃| rather than a flat ±1 has two desirable properties:\n\n'
        '  1.  Confidence weighting.  Cases where all six frontier models agree strongly\n'
        '      produce large |δ̃| and thus stronger gradient updates.  Ambiguous cases\n'
        '      (where δ̃ ≈ 0) contribute little regardless of the model\'s answer — appropriate\n'
        '      since the "correct" answer is itself uncertain.\n\n'
        '  2.  Natural scale.  The reward magnitude is tied to the utility difference, not\n'
        '      an arbitrary constant, keeping the signal stationary across goods.\n\n'
        'Responses that cannot be parsed as "Yes" or "No" are treated as irrational\n'
        'and receive −|δ̃|.'
    )); y -= 0.245

    title_text(ax, y, '5.4  Why Not Use a Symmetry Reward?', size=11, color=TEXT, bold=False); y -= 0.038
    body(ax, y, (
        'An alternative reward of +1 for consistent X/Y choices and −1 for inconsistent\n'
        'was considered and rejected.  A model that says "No" from BOTH perspectives scores\n'
        '+1 while remaining deeply irrational (high η, high λ).  The utility-weighted reward\n'
        'correctly penalises this because refusing to trade for a better good receives\n'
        '−|δ̃| regardless of cross-perspective consistency.'
    ))

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — GRPO Algorithm
# ═══════════════════════════════════════════════════════════════════════════
def page4(ax):
    y = 0.96
    title_text(ax, y, '6  GRPO Training Algorithm', size=13); y -= 0.042
    hline(ax, y); y -= 0.022

    title_text(ax, y, '6.1  Background', size=11, color=TEXT, bold=False); y -= 0.035
    body(ax, y, (
        'GRPO (Group Relative Policy Optimization, Shao et al. 2024) is a policy-gradient\n'
        'algorithm that eliminates the learned value network required by PPO.  This is\n'
        'appropriate here because our reward is a clean verifiable signal that does not\n'
        'require a separate critic to estimate.'
    )); y -= 0.105

    title_text(ax, y, '6.2  Algorithm', size=11, color=TEXT, bold=False); y -= 0.038

    # algorithm box
    algo = (
        'For each training prompt qᵢ:\n\n'
        '  Step 1.  Sample G = 16 completions {o₁, …, o₁₆} from policy π_θ at temp = 1.5\n\n'
        '  Step 2.  Score each: ℛⱼ = ℛ(oⱼ, δ̃ᵢ)   for j = 1, …, G\n\n'
        '  Step 3.  Zero-diversity group: if all G completions are identical,\n'
        '           task rewards are set to 0  (the group is NOT skipped)\n\n'
        '  Step 4.  Compute advantages:  Aⱼ = ℛⱼ − mean(ℛ)\n'
        '           (mean-centred; no std normalisation)\n\n'
        '  Step 5.  Clipped policy-gradient loss (DAPO variant):\n\n'
    )
    ax.add_patch(FancyBboxPatch((0.04, y-0.44), 0.92, 0.44,
                                boxstyle='round,pad=0.012',
                                facecolor=BOX_BG, edgecolor=BOX_BD,
                                linewidth=1, transform=ax.transAxes))
    ax.text(0.07, y-0.015, algo, transform=ax.transAxes,
            fontsize=9, color=TEXT, va='top',
            fontfamily='monospace', linespacing=1.6)

    math(ax, y-0.315,
         r'$\mathcal{L}_\mathrm{GRPO} = -\frac{1}{G}\sum_{j=1}^{G} \min\!\left(\frac{\pi_\theta(o_j|q_i)}{\pi_{\theta_\mathrm{old}}(o_j|q_i)}A_j,\; \mathrm{clip}\!\left(\frac{\pi_\theta}{\pi_{\theta_\mathrm{old}}}, 1{-}\varepsilon, 1{+}\varepsilon\right)A_j\right)$',
         size=9.5)

    ax.text(0.07, y-0.365,
            '  Step 6.  Add KL penalty:   ℒ = ℒ_GRPO + β · KL(π_θ ‖ π_ref)\n\n'
            '  Step 7.  Update θ via gradient descent',
            transform=ax.transAxes, fontsize=9, color=TEXT, va='top',
            fontfamily='monospace', linespacing=1.6)
    y -= 0.46

    title_text(ax, y, '6.3  Zero Task-Reward Advantage Groups', size=11, color=TEXT, bold=False); y -= 0.035
    body(ax, y, (
        'Because the base model says "No" 99% of the time, most early groups of G = 16\n'
        'completions are entirely "No".  Mean-centring a constant reward vector gives\n'
        'Aⱼ = 0 for all j, so the group carries no policy-gradient signal (~80% of steps).\n'
        'Nothing is skipped: stock TRL GRPOTrainer still generates, backprops and steps,\n'
        'and at β = 0.04 a zero task advantage still leaves a KL-only update.  Zeroing the\n'
        'rewards changes the LOGGED reward, not the gradient.'
    )); y -= 0.135

    title_text(ax, y, '6.4  Hyperparameters', size=11, color=TEXT, bold=False); y -= 0.038
    table(ax, y,
          ['Parameter', 'Value', 'Rationale'],
          [['Group size G',         '16',        'Handles 99% No base rate'],
           ['Temperature',          '1.5',       'Must be ≥ 1.0; lower temp worsens peaked dist.'],
           ['Clip ε',               '0.2',       'Standard PPO clip'],
           ['KL coefficient β',     '0.04',      'Tighter than DeepSeek-R1; stays near base model'],
           ['Learning rate',        '1 × 10⁻⁶', 'Conservative'],
           ['LoRA rank',            '16',        '~0.5% of params updated'],
           ['max_completion_length','4',         'Yes/No needs 1–2 tokens; 4 gives slack'],
           ['loss_type',            'dapo',      'Eliminates length bias for 1-token outputs'],
           ['scale_rewards',        'none',      'No group-std normalisation; mean-centring still applies']],
          [0.26, 0.18, 0.51], row_h=0.036)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — Infrastructure + Results
# ═══════════════════════════════════════════════════════════════════════════
def page5(ax):
    y = 0.96
    title_text(ax, y, '7  Training Infrastructure', size=13); y -= 0.042
    hline(ax, y); y -= 0.022
    body(ax, y, (
        '  • Hardware:   1 × NVIDIA A100-40GB SXM on NSCC ASPIRE 2A\n'
        '  • Base model: Qwen2.5-7B-Instruct with LoRA (rank 16) — ~0.5% of parameters updated\n'
        '  • Library:    TRL 1.3.0 GRPOTrainer + PEFT\n'
        '  • Training:   98,900 steps (one full epoch over remaining_goods.json, 49,450 cases)\n'
        '  • Wall time:  ~7 days across three sequential 72-hour PBS jobs with checkpoint resume\n'
        '  • Speed:      ~6–8 seconds/step using plain HF generate (no vLLM)'
    )); y -= 0.16

    title_text(ax, y, '8  Evaluation', size=13); y -= 0.042
    hline(ax, y); y -= 0.022
    body(ax, y, (
        'The fine-tuned model is evaluated on held-out test_goods.json (9,890 cases).\n'
        'For each case and each perspective, teacher-forced log-probability scoring is used:\n'
    )); y -= 0.07
    math(ax, y, r'$P(\mathrm{Yes}\mid\mathrm{prompt}) \propto \exp\!\left(\log p_\theta(\text{``Yes''}\mid\mathrm{prompt})\right)$',
         size=11); y -= 0.055
    body(ax, y, (
        'This gives exact normalised probabilities without sampling noise, matching the format\n'
        'used for the pre-training baseline (λ̂_before).  Choice probabilities are then fed into\n'
        'the Model A (NLS) structural estimator to jointly recover λ, η, and all item/attribute\n'
        'fixed effects.'
    )); y -= 0.115

    title_text(ax, y, '9  Results', size=13); y -= 0.042
    hline(ax, y); y -= 0.022

    # results table — parameters
    body(ax, y, 'Structural parameter estimates:'); y -= 0.045
    y = table(ax, y,
              ['Parameter', 'Before GRPO', 'SE', 'After GRPO', 'SE'],
              [['λ (loss aversion)', '11.75', '1.22', '0.177', '0.005'],
               ['η (status-quo bias)', '1.52', '0.09', '−0.048', '0.024']],
              [0.30, 0.17, 0.10, 0.17, 0.10], row_h=0.038)
    y -= 0.02
    body(ax, y, 'Human benchmark (Kahneman & Tversky 1992): λ ≈ 2.25'); y -= 0.05

    # results table — raw rates
    body(ax, y, 'Raw choice rates:'); y -= 0.045
    y = table(ax, y,
              ['Metric', 'Before GRPO', 'After GRPO'],
              [['Yes rate (X-perspective)',     '0.5%',  '44.6%'],
               ['Yes rate (Y-perspective)',     '0.5%',  '48.7%'],
               ['Fraction (No, No) cross-tab', '99.1%', '7.8%'],
               ['Fraction (No_X, Yes_Y)',       '<1%',   '47.3%']],
              [0.50, 0.20, 0.20], row_h=0.038)
    y -= 0.025

    # headline result box
    ax.add_patch(FancyBboxPatch((0.05, y-0.135), 0.90, 0.135,
                                boxstyle='round,pad=0.015',
                                facecolor='#e8f8e8', edgecolor='#2a7a2a',
                                linewidth=2, transform=ax.transAxes))
    ax.text(0.5, y-0.025,
            'λ dropped from 11.75 → 0.177  (98.5% reduction)',
            transform=ax.transAxes, fontsize=13, color=GREEN,
            fontweight='bold', ha='center', va='top')
    ax.text(0.5, y-0.065,
            'Well below the human benchmark of λ ≈ 2.25\n'
            'A 7B model fine-tuned with GRPO achieves lower loss aversion\n'
            'than frontier models 10–100× its size.',
            transform=ax.transAxes, fontsize=9.5, color='#1a4a1a',
            ha='center', va='top', linespacing=1.6)

# ═══════════════════════════════════════════════════════════════════════════
# ASSEMBLE PDF
# ═══════════════════════════════════════════════════════════════════════════
pages = [page1, page2, page3, page4, page5]
tmp_files = []

for i, fn in enumerate(pages):
    fig = make_section_fig(f'page{i+1}', fn)
    tmp = fig_to_tmp(fig, dpi=200)
    tmp_files.append(tmp)
    print(f'  rendered page {i+1}')

pdf = FPDF(unit='mm', format='A4')
for tmp in tmp_files:
    pdf.add_page()
    pdf.image(tmp, x=0, y=0, w=210, h=297)

out = os.path.join(os.path.dirname(__file__), 'training_overview.pdf')
pdf.output(out)
print(f'\nSaved: {out}')

for tmp in tmp_files:
    os.unlink(tmp)
