# Figure sources

The overview, CASO architecture, replay protocol, and runtime chart are PNG renderings of the manuscript PDFs, with no changes to their content. The acceptance comparison is redrawn from the current manuscript table. The manuscript source remains outside the public release.

| Public asset | Local source | Source SHA-256 |
| --- | --- | --- |
| `mosaic-overview.png` | `paper/figure/MOSAIC.pdf` | `42785947e5ac31258d04019b4a22ad8c3cd9aacdb4a699337e1df05af65a084e` |
| `caso-architecture.png` | `paper/figure/CASO.pdf` | `90ba878eda030d338d3898d046ba8eb51a568d54f1efb22c3429106907a2d087` |
| `promosim-protocol.png` | `paper/figure/PromoSim.pdf` | `0baa40f7afdcb35437a7fb314cb7f1fc50d5752538105fd9c0cf0842ddbc338f` |
| `policy-runtime.png` | `paper/figure/isr/fig23_runtime_comparison.pdf` | `ad049cf88b3e241b2ee2ec2979648e9e4a1ee92a08b5f66f62fa1efa0d9c72fc` |

To regenerate the PDF renderings from the local manuscript with Poppler:

```bash
pdftoppm -f 1 -singlefile -scale-to 1800 -png paper/figure/MOSAIC.pdf assets/mosaic-overview
pdftoppm -f 1 -singlefile -scale-to 2000 -png paper/figure/CASO.pdf assets/caso-architecture
pdftoppm -f 1 -singlefile -scale-to 2000 -png paper/figure/PromoSim.pdf assets/promosim-protocol
pdftoppm -f 1 -singlefile -scale-to 1700 -png paper/figure/isr/fig23_runtime_comparison.pdf assets/policy-runtime
```

## Acceptance comparison

`acceptance-comparison.png` is a grouped bar chart of the current values in
`paper/ISRE-template.tex`, table `tab:aligned-all-methods`. It compares CASO
with the strongest baseline selected separately for each reporting cell.
Values are mean predicted acceptance over three reporting movies. The local
provenance record, `experiments/results/scalar_result_provenance.json`, identifies
these as CASO scalar predictions trained on retained PromoSim outcomes, with no
new policy replays. The plot therefore labels them as predictions rather than
independent replay measurements.

| Setting | Budget | CASO | Strongest baseline | Baseline mean |
| --- | ---: | ---: | --- | ---: |
| ID | 5 | 241.8 | ClusterRank | 226.4 |
| ID | 10 | 322.0 | S2V-DQN (IC) | 306.6 |
| Zero-shot OOD | 5 | 219.7 | ClusterRank | 213.0 |
| Zero-shot OOD | 10 | 303.2 | GCOMB (IC) | 287.6 |

The chart was rendered with Matplotlib on a 12 × 5.5 inch white canvas at
180 dpi, with a shared zero-based y-axis. It adds no new experimental results
or uncertainty estimates.
