# RGB-thermal VPR ranking audit

The author's analysis code is available under the MIT license in LICENSE. This permission does not cover third-party numerical inputs or example images; see NOTICE.md. Public numerical archiving and a version-specific DOI are still pending, so this code alone does not establish complete public reconstruction.

The versioned numerical archive supplies frozen distances, labels, original candidate identities, saved sampling weights, archived predictions and expected/reference evidence. Extract the complete archive into a separate folder and use the entrypoint shipped in that folder:

```text
python -m pip install -r requirements.txt
python -X utf8 -B reproduce.py --input-root THIS_DIRECTORY --output-root NEW_OUTPUT --audit-workers 4
```

The new output must not exist. Python 3.11.9 and NumPy 2.4.4 were used for local verification. No feature extraction, model training, new data sampling or external inference is run. Worker count only schedules independent exact checks. Integer identities and certificates are exact; finite result comparisons use rtol=0 and atol=1e-11. Printed cells are checked at their stated precision. Legacy metadata-coordinate joins retain their original identity-join tolerance and use frozen geographic distances for actual correctness.

Code in this directory must agree byte-for-byte with the corresponding code in the fixed Zenodo version. The full archive additionally locks each input and code member. `CODE_MANIFEST.json` records the file hashes and the associated numerical archive hash.

Code and manuscript author: Shuo Yu (corresponding author), School of Electronic Information and Electrical Engineering, Chengdu University. Project: 2026 Undergraduate Innovation Training Program of Chengdu University, No. 202611079009, Trustworthy Localization and Safe Autonomous Decision-Making for Embodied Mobile Robots in Complex Environments.

Outputs distinguish numerical recalculation, historical-prediction aggregation, and documentary/image source verification. A source-bound historical control is not a rerun of its training or matching procedure. The result is not raw-image-to-result reproduction or independent external-dataset replication.

