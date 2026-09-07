# RSNA Knee Abnormality Detection — Competition Description

> **Single source for this document:** <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview>
> (*Overview*, *Evaluation*, *Timeline*, *Prizes*, *Code Requirements*,
> *Efficiency Prize* and *Data* tabs), retrieved 7 September 2026.
>
> This file is a **transcription of the official page**. It contains no analysis of
> our own — our measurements and interpretations live in [`data.md`](data.md). If
> something here is wrong, it means the page changed on Kaggle: resync it, don't
> patch it by hand.

---

## Overview

> A single knee scan can reveal a dozen different problems. In this competition,
> you are tasked to build machine learning models that detect a defined set of
> clinically important abnormalities on knee MRI examinations.

## Description

> The knee is the most commonly injured and imaged joint in the body.
> Osteoarthritis alone affects an estimated 654 million people worldwide, while
> acute knee injuries account for 15 to 40 percent of all sports-related trauma.
> MRIs show clinicians ligaments, cartilage, menisci, and bone in detail, without
> exposing patients to radiation.
>
> Reading those scans isn't always straightforward. ACL and MCL tears, meniscal
> damage, cartilage loss, fractures, and other abnormalities can be subtle, and
> radiologists don't always interpret them the same way. Access to musculoskeletal
> radiologists is also limited, especially outside major medical centers, leading
> to delays and inconsistent diagnoses.
>
> In this competition, you will develop multimodal machine learning models to
> detect twelve clinically important knee abnormalities. You'll work with the first
> RSNA AI Challenge dataset that pairs every imaging study with its original
> radiology report, enabling your models to learn from both visual scans and
> written diagnostic text.
>
> High-performing models can act as robust decision support tools, delivering the
> accuracy, consistency, and speed needed to elevate expert-level knee MRI
> interpretation and improve care across disparate clinic settings.

---

## Evaluation

Submissions are evaluated by the average area under the ROC curve between the
predicted confidence scores and the observed targets across the twelve targets:

$$\text{Final Score} = \frac{1}{12} \sum_{i=0}^{11} \text{AUC}_i$$

The final score is, in other words, the **macro-averaged AUC ROC**.

### Submission File

For each row in the test set, you must predict a confidence score for each of the
twelve target labels. The file should contain a header and have the following
format:

```csv
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid_1>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
<uid_2>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
...
```

The submission file must be named **`submission.csv`**.

---

## Timeline

| Date (2026) | Milestone |
|---|---|
| July 30 | **Start Date** |
| **October 15** | **Entry Deadline.** You must accept the competition rules before this date in order to compete. |
| **October 15** | **Team Merger Deadline.** This is the last day participants may join or merge teams. |
| **October 22** | **Final Submission Deadline** |
| November 5 | **Winners' Requirement Deadline.** Deadline for winners to submit to the host/Kaggle their training code, video and method description. |

> All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise
> noted. The competition organizers reserve the right to update the contest
> timeline if they deem it necessary.

---

## Prizes

**Total prize pool: $77,000.**

### Main Leaderboard

| Rank | Prize |
|---|---|
| First | $9,000 |
| Second | $7,000 |
| Third | $6,500 |
| Fourth | $6,000 |
| Fifth | $5,500 |
| Sixth | $5,000 |
| Seventh | $5,000 |
| Eighth | $5,000 |
| Ninth | $5,000 |
| Tenth | $5,000 |

### Efficiency Track

| Rank | Prize |
|---|---|
| First Efficiency Prize | $7,000 |
| Second Efficiency Prize | $6,000 |
| Third Efficiency Prize | $5,000 |

> Because this competition is being hosted in coordination with the Radiological
> Society of North America (RSNA) Annual Meeting, winners will be invited and
> strongly encouraged to attend the AI Challenge Recognition Event with waived fee,
> contingent on review of solution and fulfillment of winners' obligations.

### Winners' Obligations

Per the competition rules, in addition to the standard Kaggle Winners' Obligations
(open-source licensing requirements, solution packaging/delivery, presentation to
host), the host team also asks that you:

1. create a short video presenting your approach and solution;
2. publish a link to your open sourced code and the weights on the competition
   forum;
3. share the final version of the model as publicly available for open distribution
   and validation. Example given:
   <https://www.kaggle.com/models/tom99763/9th-place-models-rsna-iad/PyTorch/default>

---

## Code Requirements

Submissions to this competition must be made through **Notebooks**. In order for
the "Submit" button to be active after a commit, the following conditions must be
met:

- **CPU Notebook ≤ 9 hours run-time**
- **GPU Notebook ≤ 9 hours run-time**
- **Internet access disabled**
- **Freely & publicly available external data is allowed, including pre-trained
  models**
- Submission file must be named **`submission.csv`**

See the *Code Competition FAQ* for more information on how to submit, and the *code
debugging doc* if you encounter submission errors.

---

## Efficiency Prize

> We are hosting a second track that focuses on model efficiency, because highly
> accurate models are often computationally heavy.

For the Efficiency Prize, submissions are evaluated on **both runtime and
predictive performance**.

### Eligibility

To be eligible for an Efficiency Prize, a submission:

- Must be among the submissions selected by a team for the Leaderboard Prize, or
  else among those submissions automatically selected under the conditions
  described in the *My Submissions* tab.
- Must be ranked on the Private Leaderboard **higher than the
  `sample_submission.csv` benchmark**.

All submissions meeting these conditions will be considered for the Efficiency
Prize. **A submission may be eligible for both the Leaderboard Prize and the
Efficiency Prize.**

An Efficiency Prize will be awarded to eligible submissions according to how they
are ranked by the following evaluation metric on the private test data. More
details may be posted via discussion forum updates.

### Efficiency Score

We compute a submission's efficiency score by:

$$\text{Efficiency} = \frac{\text{AUC}}{\text{Benchmark} - \max \text{AUC}} + \frac{\text{RuntimeSeconds}}{32400}$$

where:

- **AUC** is the submission's score on the main competition metric;
- **Benchmark** is the score of the benchmark `sample_submission.csv`;
- **max AUC** is the maximum AUC of all submissions on the Private Leaderboard;
- **RuntimeSeconds** is the number of seconds it takes for the submission to be
  evaluated.

**The objective is to minimize the efficiency score.**

During the training period of the competition, a leaderboard for the *public* test
data is available in a notebook (*Efficiency Leaderboard*), updated daily. After
the competition ends, that leaderboard will be updated with efficiency scores on
the *private* data. During the training period, it shows **only the rank of each
team, not the complete score**.

---

## Dataset Description

> This dataset contains knee MRI studies annotated for twelve common findings:
> ligament and meniscus injuries, three compartments of osteoarthritis, joint
> effusion, synovitis, Baker's cyst, bone contusion, and fracture. Each study
> comprises a collection of individual MRI sequences from a single scanning session
> formatted as DICOM series. Your task is to predict the per-study probability of
> each of the twelve findings.
>
> Studies come from a diverse international mix of imaging sites and span a wide
> range of scanners, protocols, and populations. **Only a small subset of training
> studies carry per-condition labels.** We also provide the original text of the
> radiology report from which you may wish to derive the labels for the remaining
> studies.

### Files

#### `train.csv` — one row per training study

| Column | Description |
|---|---|
| `StudyInstanceUID` | Unique identifier for the study; matches the folder name under `train_series/` |
| `Report` | The free-text radiology report. **May be in any of several languages**, depending on the reporting institution |

Plus **twelve binary labels (0/1)**:

| Column | Meaning |
|---|---|
| `ACL` | Anterior cruciate ligament injury |
| `MCL` | Medial collateral ligament injury |
| `Medial Meniscus` | Medial meniscus tear |
| `Lateral Meniscus` | Lateral meniscus tear |
| `Medial OA` | Osteoarthritis of the medial tibiofemoral compartment |
| `Lateral OA` | Osteoarthritis of the lateral tibiofemoral compartment |
| `PF OA` | Patellofemoral osteoarthritis |
| `Effusion` | Joint effusion / excess fluid |
| `Synovitis` | Inflammation of the joint lining |
| `Baker's` | Baker's cyst |
| `Contusion` | Bone contusion / bone bruise |
| `Fracture` | Fracture |

#### `train_series.csv` — one row per training series

Each series is a single MRI acquisition, and each study comprises several series.

| Column | Description |
|---|---|
| `StudyInstanceUID` | Study this series belongs to |
| `SeriesInstanceUID` | Unique identifier for the series; matches the folder name under `train_series/<StudyInstanceUID>/` |
| `Fluid_Sensitive` | 1 if the sequence emphasizes fluid signal (T2, PD, STIR, and similar), 0 otherwise |
| `Fat_Suppression` | 1 if the sequence applies fat suppression, 0 otherwise |
| `Anatomical_Plane` | Imaging plane: `Sagittal`, `Coronal`, or `Axial` |

> ⚠️ **Official note on these two columns:** "although `Fluid_Sensitive` and
> `Fat_Suppression` are often correlated, as observed in the training set, **they
> are not necessarily equivalent for every case**." See [`data.md`](data.md): in the
> CSVs we have, they are identical on 100% of rows — but do not write code that
> assumes this will hold at test time.

#### `train_series/` — training DICOMs

Organized as
`train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`.

Each `.dcm` is **a single image slice**. Series typically contain **20–45 slices
(median 30)**, with a long tail out to a few hundred.

#### `test.csv`

Example test file with three study IDs from the public test set. During scoring,
this example data will be replaced with the actual test data.

> **There are about 1,300 studies in the test set.**
>
> ⚠️ **The `Report` field will not be provided at the testing stage.**

| Column | Description |
|---|---|
| `StudyInstanceUID` | Unique identifier for a test study |

#### `test_series.csv`

Same schema as `train_series.csv`, for the example test studies. Replaced with the
real test-series descriptors during scoring.

#### `test_series/`

Example test DICOMs, same layout as `train_series/`. Replaced with the real test
DICOMs during scoring.

#### `sample_submission.csv`

A valid submission with all label columns set to `0.5`.

### Dataset Distribution Notice

> Although efforts have been made to ensure each abnormality is represented in each
> dataset, **the prevalence of abnormalities is not guaranteed to be the same
> across the training, public leaderboard, and final evaluation datasets.**

### DICOM Notes

- **Intensities, orientations, and resolutions vary** across series and studies.
- Series come in a **mix of transfer syntaxes**: uncompressed Explicit VR Little
  Endian, JPEG Lossless, JPEG 2000, Implicit VR Little Endian.
- Every DICOM has been **stripped to an allowlisted set of 86 metadata tags**.

---

## Acknowledgements

> RSNA would like to thank the following individuals and organizations whose
> contributions made possible the RSNA Knee Abnormality Detection AI Challenge.

### Challenge Organizing Team

- Po-Hao "Howard" Chen, MD, MBA — Cleveland Clinic, USA
- Naveen Subhas, MD, MPH — Cleveland Clinic, USA
- Oganes Ashikyan, MD — UT Southwestern, USA
- Pieter Baeyens, MD — AZ Delta, Belgium
- Robyn Ball, PhD — The Jackson Laboratory, USA
- Errol Colak, MD — Unity Health Toronto, University of Toronto, Canada
- Ali Emami, PhD — Emory University, USA
- Adam Flanders, MD — Thomas Jefferson University, USA
- Hillary Garner, MD — Mayo Clinic Jacksonville, USA
- Jacob Kazam, MD — Cornell University, USA
- Felipe Kitamura, MD, PhD — Universidade Federal de São Paulo, Brazil
- Hui-Ming Lin, HBSc — Unity Health Toronto, Canada
- Luciano Prevedello, MD, MPH — Ohio State University, USA
- Daniel Schneider, MD — Cleveland Clinic, USA
- Paul Yi, MD — St. Jude Children's Research Hospital, USA

### Data Contributors

> Thank you to the following institutions for contributing de-identified MRI
> images, radiology reports and associated clinical data that was assembled to
> create the challenge dataset:

- AZ Delta, Roeselare, Belgium
- Centro Rossi, Buenos Aires, Argentina
- Chiang Mai University, Chiang Mai, Thailand
- China Medical University Hospital, Taichung, Taiwan
- CHU Mohamed VI Cadi Ayyad University, Marrakech, Morocco
- Clinica Alemana Santiago de Chile, Santiago, Chile
- Hacettepe University School of Medicine, Ankara, Türkiye
- Khon Kaen University, Khon Kaen, Thailand
- Koç University Hospital, Istanbul, Türkiye
- Mater Dei Hospital, Msida, Malta
- McGill University Health Centre, Montreal, Canada
- Samsun Training and Research Hospital, Samsun, Türkiye
- Sofia University "St. Kliment Ohridski", Sofia, Bulgaria
- Thomas Jefferson Hospital, Philadelphia, PA, USA
- Unity Health Toronto, Toronto, Canada
- University Hospital Dubrava, Zagreb, Croatia
- University Hospital of Heraklion, Crete, Greece
- University Hospital of Würzburg, Würzburg, Germany
- University of Sarajevo, Sarajevo, Bosnia and Herzegovina

Additional contributing sites:

- Intermed Hospital, Ulaanbaatar, Mongolia
- Liverpool Hospital, Liverpool, NSW, Australia
- Salus Vigevano Centro Sanitario Depa, Vigevano, Italy

### Data Curators

- Hui-Ming Lin, HBSc — Unity Health Toronto, Canada
- Jason Sho — RSNA, USA

### Data Annotators

> The challenge organizers wish to thank the Society of Skeletal Radiology and the
> International Skeletal Society for recruiting its members to join the annotation
> team that labeled the dataset used in the challenge.

- Pieter VanDyck, MD, PhD — University Hospital Antwerp, Belgium
- Nicholas Marc Beckmann, MD — UTHealth – McGovern School of Medicine, USA
- Takeshi Fukuda, MD, PhD — The Jikei University School of Medicine, Japan
- Hiroshi Yoshioka, MD, PhD — University of California, Irvine, USA
- Jee Won Chai, MD, PhD — SMG-SNU Boramae Medical Center, Republic of Korea
- Kathryn J. Stevens, MB, BS — Stanford University School of Medicine, USA
- Kevin C. McGill, MD, MPH — University of California, San Francisco, USA
- Christopher J. Gottsegen, MD — NYU Grossman School of Medicine, USA
- Joseph Tang, MD — University of Wisconsin, USA
- Debajyoti Saha, MD — UMass Memorial Medical Center, USA
- Parthiv N Mehta, MBBS, MD (DABR) — Virtual Radiologic, Inc, USA
- Youngjune Kim, MD, PhD — Seoul National University Bundang Hospital, Republic of Korea
- Pamela J. Walsh, MD — Northwell Health, USA
- Jason Matakas, MD — Weill Cornell Medical College / New York Presbyterian Hospital, USA
- Daniel M. Walz, MD — Lenox Hill Hospital / Northwell Health, USA
- Matthew Irwine, MD — Mayo Clinic, USA
- Michael Hoy, MD — Thomas Jefferson University Hospital, USA
- Tatiane Cantarelli Rodrigues, MD — Ottawa Hospital, University of Ottawa, Canada
- Gregory Dave R. Taverner, MD, FPCR, FCTMRISP, FUSP — St. Luke's Medical Center, Philippines

### Report and Image QC Reviewers

> Thank you to the following individuals for their contributions to the quality
> review of radiology reports and imaging data.

- Lejla Aganovic, MD — University of California, San Diego, USA
- Ersa Akcicek, MD — Lunenfeld Tanenbaum Research Institute, Canada
- Reza Al-Saudi
- Ferco Berger, MD, FRCPC — Sunnybrook Health Sciences Centre, University of Toronto, Canada
- Rodrigo Borrero-Leon, MD — Fundación Cardioinfantil-LaCardio, Colombia
- Jason Ciotola-Koch, DO — USA
- Ceylan Colak, MD — Mayo Clinic, USA
- Priscila Crivellaro, MD — St. Michael's Hospital, University of Toronto, Canada
- Susanne Gaube, PhD — UCL Global Business School for Health, University College London, United Kingdom
- Violeta Groudeva, MD, PhD — University Hospital Saint Ekaterina, Medical University Sofia, Bulgaria
- Samir Grover, MD, MEd, FRCPC — University of Toronto, Canada
- Sebastiaan Hermans, MD — Heilig Hart Ziekenhuis, Belgium
- Jeffrey D. Jaskolka, MD, FRCPC — University of Toronto, Canada
- Markus Lammle, MD, PhD — Upstate Medical University, USA
- Lara Gabrielle Lim, MD — Unity Health, Canada
- Muhammad Munshi, MD, FRCPC — University of Toronto, Canada
- Anastasia Oikonomou, MD, PhD, FRCPC — Sunnybrook Health Sciences Centre, University of Toronto, Canada
- Dawn Pearce, MD — Unity Health, Canada
- Samia Sayyid
- Andreas Schicho, MD, EDIR, EBIR — Germany
- Senad Senderovic — Canada
- Rafael Boava Souza, MD — Universidade Federal de São Paulo (UNIFESP), Brazil
- Monica Tafur, MD, FRCPC — St. Michael's Hospital, University of Toronto, Canada
- Paraskevi A. Vlachou, MBChB, FRCR — Unity Health Toronto, Canada
- Sahika Betul Yayli, MD — Mayo Clinic, USA
- Ali Yikilmaz, MD — McMaster University, Canada

> Special thanks to **MD.ai** for providing tooling for the data annotation process.

---

## Citation

> Po-Hao "Howard" Chen, Naveen Subhas, Robyn Ball, Pieter Baeyens, Errol Colak,
> Ali Emami, Hillary Garner, Jacob Kazam, Hui-Ming Lin, Luciano Prevedello,
> Daniel Schneider, Jason Sho, Ryan Holbrook, and María Cruz. RSNA Knee Abnormality
> Detection. <https://kaggle.com/competitions/rsna-knee-abnormality-detection>,
> 2026. Kaggle.
