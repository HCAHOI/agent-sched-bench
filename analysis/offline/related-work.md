# Related-work comparison

This optional note records the comparison used by the current claims. It is not
a project-status document. Paper versions checked for this comparison were
Continuum arXiv:2511.02230v6, ThunderAgent arXiv:2602.13692v3, TraceLab
arXiv:2606.30560v2, CacheSage arXiv:2605.27744, and PEEK arXiv:2607.02525.

## Continuum

Continuum selects a per-tool-name KV-cache TTL from an empirical duration CDF.
Its benefit term includes a sliding-window average queue-delay signal, so it is
partially load-aware; describing it as load-blind is incorrect. The load term is
workload-level rather than an instantaneous per-request memory price.

The closest contrast is therefore conditional-residual expected-cost pricing
versus a frozen tool-name point TTL, not “load awareness versus no load
awareness.” Continuum's published `cd` tail illustrates why tool identity can be
too coarse for compound commands; the future Continuum experiment must test
whether command-prefix conditioning improves decision utility.

## ThunderAgent

ThunderAgent is program-aware scheduling rather than a richer duration
predictor. Its program state includes identity, context length, tool environment,
placement, phase, and status. The scheduler uses context length and an
elapsed-time decay under a memoryless assumption; it does not require a known
program DAG or next-tool predictor.

The evaluated version has public code and Dynamo/SkyRL integration. Any
head-to-head must extract and match the scheduler honestly rather than compare
uncontrolled stacks.

ThunderAgent's reported failure of indiscriminate PCIe swapping motivates
selective offload, but it is not evidence that this project succeeds. The
project must still measure transfer contention and serving effects live at the
fixed operating point.

## Position of this work

The estimator spectrum is:

- memoryless elapsed-time decay;
- tool-name empirical duration distribution;
- command-prefix conditional residual-time prior;
- a per-workload utility certificate deciding whether the conditioned policy is
  authorized against its fallback.

The distinctive claim is the last step: decision utility at the operating point
decides what ships. The WTN failure demonstrates why estimator fit alone is not
enough. The offline certificate does not replace live systems evaluation and
does not certify later adaptive states.

## Adjacent actions

CacheSage is the closest survival-based eviction/prefetch comparison. PEEK and
ThunderAgent reinforce that memory pressure and transfer interference must be
measured in the serving stack. TraceLab is a useful third-party workload source,
but any use must preserve whole-program order and an unopened evaluation
boundary.

Must-cite methodological context includes Lindon et al. (KDD 2022), Dinitz et
al. arXiv:2202.09312 on prediction portfolios, and AdaSwitch arXiv:2509.02302.
