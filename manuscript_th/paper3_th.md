# Paper 3 (`ucai-sovereign-box`): ระบบสืบค้นวรรณกรรมแบบออฟไลน์บนการ์ดจอเดี่ยว การตรวจอ้างอิง-ความปลอดภัย และการรับรองความซ้ำด้วย RO-Crate

> **ชื่อเรื่องภาษาอังกฤษ:** *`ucai-sovereign-box`: Single-GPU Air-Gapped Literature Retrieval, Citation-Security Auditing, and RO-Crate Reproducibility*
> **ผู้วิจัย:** นพดล ชัยวิสูตร (Noppadon Chaiwisut) — นักวิจัยอิสระ ประเทศไทย (`Independent Researcher, Thailand`)
> **ORCID:** `0009-0005-2881-8458` | **อีเมล:** `noppadon.chaiwisut@gmail.com`
> **รุ่น / วันที่:** Version 0.2.0 (Tier-2 Full-Corpus Single-GPU Release) | พฤศจิกายน 2026
> **หมายเหตุการทำซ้ำ:** โค้ด (`MIT`) & `UCAI-SovereignBox-Bench-v2` (`CC BY 4.0`); Colab: `1gJZ47SwcbsV5GCfdjTDWkHw4rRVF7j25`; Zenodo DOI: *Pending v0.2.0 Release*

---

## บทคัดย่อ (Abstract)

ผู้ช่วยวิจัย AI บนคลาวด์เชิงพาณิชย์สร้างอุปสรรคเชิงโครงสร้าง 3 ประการ ได้แก่ ค่าใช้จ่าย Token ต่อเนื่องและปัญหา Rate-Limit, ความเสี่ยงด้านอธิปไตยข้อมูลเมื่อส่งต้นฉบับที่ยังไม่ตีพิมพ์ออกนอกองค์กร, และสถานะไฟล์ที่ไม่สามารถทำซ้ำได้

เราเสนอ **`ucai-sovereign-box` (v0.2.0)** ระบบ *Research-in-a-Box* แบบออฟไลน์บนการ์ดจอเดี่ยว ที่รวม:
- การสืบค้นลูกผสม Inverted Okapi BM25 + `all-MiniLM-L6-v2`
- การตรวจแหล่งที่มาตัวเลขข้ามเอกสารพร้อมกลไกงดเว้น `DOC_NOT_INDEXED` (Spoke 1)
- การล้างคำสั่งแฝงโจมตี (Spoke 2)
- ตัวเลือกซ้อนโมเดลภาษาเปิดในเครื่อง (`Qwen2.5-1.5B-Instruct`)
- การแพ็กหลักฐานตรวจสอบตัวเองด้วย **`RO-Crate 1.1` + `SHA-256`** ที่ **ต้นทุนคลาวด์ `$0.00`**

การทำดัชนีครบทั้ง **`168` Open-Access PDFs** (`17,623` ชิ้นข้อมูลรายหน้า, `80,593` คำศัพท์ BM25, `0` ไฟล์ตกหล่น) บนการ์ดจอ `NVIDIA L4` ใช้ VRAM เพียง **`409.99 MB`** (`1.82%` ของ L4 / `2.50%` ของ T4 16 GB) บนชุดทดสอบ **`UCAI-SovereignBox-Bench-v2`** (`N = 100` งาน) ตัวสืบค้น `BM25-Only` และ `Dense-Only` ที่ไร้เกราะป้องกันได้ `Safety F1 = 0.0000` ขณะที่ `ucai-sovereign-box` ทำได้ **`0.7400 Hit@3`**, **`0.9200 Safety Accuracy`**, และ **`0.9184 Safety F1`** (`95% CI [0.8542, 0.9677]`; McNemar `45:3` เทียบกับ BM25-Only, `p < 10⁻⁸`) ที่ความเร็ว **`25.05 ms/งาน`** เมื่อซ้อน `Qwen2.5-1.5B-Instruct` (FP16) บน GPU เดียวกัน ใช้ VRAM รวมเพียง **`3,086.07 MB`** (`13.68%` ของ L4 / `~18.8%` ของ T4) ที่ **`229.2 ms/งาน`** (เร็วกว่าผู้ตรวจบนคลาวด์ `35×`) พร้อมตรวจยืนยัน `RO-Crate 1.1` ครบ **`6/6` ไฟล์ (`100%`)**

---

## 1–2. บทนำและสถาปัตยกรรมระบบ (`ucai-sovereign-box` v0.2.0)

1. **Inverted Hybrid Index (`sovereign_index.py`):** สกัดข้อความซ้อนทับขนาด `750` ตัวอักษรผ่าน `PyMuPDF` จากทั้ง `168` PDFs (สูงสุด `20` หน้า / `120` chunks ต่อเล่ม) รวมคะแนน `S(q, c) = 0.55 · BM25(q, c) + 0.45 · max(0, cos(eq, ec))`
2. **Spoke 1/2 Deterministic Guard (`evaluator.py`):** ดักจับ Prompt Injection ครบทุกรูปแบบ, บังคับ `DOC_NOT_INDEXED_ABSTAIN` เมื่อไม่มีเอกสารในดัชนี, และตรวจตัวเลขเทียบกับเอกสารที่อ้างและอันดับผู้บริจาค (Donor ranks)
3. **Co-Located Open-Weight LLM (`Qwen2.5-1.5B-Instruct`):** โหลดแบบ FP16 บนการ์ดจอเดียวกันเพื่อตรวจข้อที่ผ่าน Stage 1 โดยไม่ต่ออินเทอร์เน็ต
4. **Cryptographic Reproducibility (`crate_packager.py`):** สร้าง `ro-crate-metadata.json` (`RO-Crate 1.1`) และ `SHA256SUMS.txt` พร้อมตรวจสอบความถูกต้องอัตโนมัติ

---

## 3. การทดลองและผลลัพธ์ (`168` PDFs, `N = 100`)

สร้างดัชนีครบ `168` ไฟล์ (`17,623` chunks, `80,593` คำศัพท์, ใช้เวลาสร้างดัชนีรวมอ่าน Drive I/O **`330.73 s`**, RAM เครื่อง **`1,780.5 MB`** บน `NVIDIA L4` `22,563.12 MB` VRAM):

### ตารางที่ 1: ผลเปรียบเทียบการ์ดจอเดี่ยวครบคลัง (`168` PDFs, `N = 100`, `$0.00` Cloud Cost)

| โหมดการทำงาน | Hit@3 `↑` | MRR@3 `↑` | Safety Acc. `↑` | F1 [95% CI] `↑` | FPR `↓` | Latency | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `BM25-Only` | `0.6900` | `0.6200` | `0.5000` | `0.0000 [0.000, 0.000]` | `0.0000` | **`13.05 ms`** | `409.99 MB` |
| `Dense-Only (MiniLM)` | `0.6200` | `0.4583` | `0.5000` | `0.0000 [0.000, 0.000]` | `0.0000` | `24.43 ms` | `409.99 MB` |
| **`UCAI Sovereign Box v0.2.0`** | **`0.7400`** | **`0.6433`** | **`0.9200`** | **`0.9184 [0.854, 0.968]`\*\*\*** | `0.0600` | `25.05 ms` | **`409.99 MB`** |
| `Sovereign + MiniLM Reranker` | `0.7000` | `0.6250` | **`0.9200`** | **`0.9184 [0.854, 0.968]`\*\*\*** | `0.0600` | `48.24 ms` | `609.30 MB` |
| `Sovereign + Qwen2.5-1.5B` | **`0.7400`** | **`0.6433`** | **`0.9200`** | **`0.9184 [0.854, 0.968]`\*\*\*** | `0.0600` | `229.2 ms` | `3,086.07 MB` |

- **วิเคราะห์ข้อผิดพลาด (`8/100` ข้อ):**
  - **False Positives `3` ข้อ (`UCAI-SOV-004, 014, 024`):** เกิดบนเอกสาร `2409.13740` (*PaperQA2*) เนื่องจากตัวเลข `15,616` อยู่เลยหน้า `20` ซึ่งเป็นเพดานการอ่านเริ่มต้น
  - **False Negatives `5` ข้อ (`UCAI-SOV-054, 058, 064, 068, 074`):** เกิดจากตัวเลขสั้น (`15`, `10`) ไปพ้องกับเลขอื่นภายในเอกสารที่อ้าง ยืนยันความจำเป็นของการอ่านครบทุกหน้าและการตรวจบริบทเชิงความหมาย Stage 2 (Paper 1)

---

## 4–5. ข้อจำกัด จริยธรรม และบทสรุป

1. เพดานการสกัดข้อความ `20` หน้า (`120` chunks) ต่อไฟล์ช่วยคุมเวลาสร้างดัชนี (วัดรอบ warm cache บน L4 ได้ `52.17 s` / รอบอ่าน Drive สด `330.73 s`) แต่ตัวเลขท้ายเล่มเกินหน้า `20` จะถูกแจ้ง `UNVERIFIED_NUMBERS` หากไม่เปิดโหมดอ่านครบทุกหน้า
2. การทดสอบ Ablation ด้วย Cross-Encoder ทั่วไป (`ms-marco-MiniLM-L-6-v2`) เพิ่ม VRAM เป็น `609.30 MB` (`3,401.6 MB` เมื่อซ้อน Qwen) และช้าลงเป็น `48.24 ms` แต่ `Hit@3` กลับลดเหลือ `0.7000` ยืนยันว่าคะแนน Hybrid BM25+Dense แบบ min-max เหมาะกับการค้นหาตัวเลขทางวิทยาศาสตร์มากกว่า
3. ด้านจริยธรรม ผลแจ้งเตือนทั้งหมดอยู่ภายใต้ **`L2 HITL`** และไม่มีการส่งข้อมูลออกนอกเครื่อง โดยสรุป `ucai-sovereign-box v0.2.0` แสดงให้เห็นว่าผู้ช่วยวิจัย AI ที่ตรวจสอบได้ครบวงจรสามารถรันบนการ์ดจอทั่วไปเครื่องเดียวได้ที่ต้นทุนคลาวด์ **`$0.00`**

---

## แหล่งอ้างอิง (References)

1. UNESCO (2021). *UNESCO Recommendation on Open Science*. https://unesdoc.unesco.org/ark:/48223/pf0000379949
2. Fischer, Markus and Meiler, Jens and Elgeti, Matthias (2026). *MuninAI: A Self-Hosted Agentic Framework for Independent Research Groups*. https://arxiv.org/abs/2610.23480
3. Pineau, Joelle and Vincent-Lamarre, Philippe and Sinha, Koustuv and Larivi\`ere, Vincent and Beygelzimer, Alina and d'Alch\'e-Buc, Florence and Fox, Emily and Larochelle, Hugo (2021). *Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)*. https://www.jmlr.org/papers/volume22/20-303/20-303.pdf
4. Kapoor, Sayash and Cantrell, Emily M. and Peng, Kenny and Pham, Thanh Hien and Bail, Christopher A. and Gundersen, Odd Erik and Hofman, Jake M. and Hullman, Jessica and Lones, Michael A. and Malik, Momin M. and Nanayakkara, Priyanka and Poldrack, Russell A. and Raji, Inioluwa Deborah and Roberts, Michael and Salganik, Matthew J. and Serra-Garcia, Marta and Stewart, Brandon M. and Vandewiele, Gilles and Narayanan, Arvind (2024). *REFORMS: Consensus-Based Recommendations for Machine-Learning-Based Science*. https://doi.org/10.1126/sciadv.adk3452
5. UK Reproducibility Network (2024). *Computational Reproducibility: A Primer from UKRN*. https://eprints.whiterose.ac.uk/id/eprint/239601/1/Computational%20Reproducibility%20Primer.pdf
6. Wilkinson, Mark D. and Dumontier, Michel and Aalbersberg, IJsbrand Jan and Appleton, Gabrielle and Axton, Myles and Baak, Arie and others (2016). *The FAIR Guiding Principles for Scientific Data Management and Stewardship*. https://pure.manchester.ac.uk/ws/files/57031836/sdata201618.pdf
7. Barker, Michelle and Chue Hong, Neil P. and Katz, Daniel S. and Lamprecht, Anna-Lena and Martinez-Ortiz, Carlos and Psomopoulos, Fotis and others (2022). *Introducing the FAIR Principles for Research Software*. https://doi.org/10.1038/s41597-022-01710-x
8. Soiland-Reyes, Stian and Sefton, Peter and Crosas, Merc\`e and Castro, Leyla Jael and Coppens, Frederik and Fern\'andez, Jos\'e Mar\'\ia and others (2022). *Packaging Research Artefacts with RO-Crate*. https://doi.org/10.3233/ds-210053
9. Gundersen, Odd Erik and Cappelen, S. and M\oln\aa (2025). *Computational Reproducibility: What to Expect from Journals and Authors*. https://ceur-ws.org/Vol-4118/paper1.pdf
10. Manning, Christopher D. and Raghavan, Prabhakar and Sch\"utze, Hinrich (2009). *Introduction to Information Retrieval*. https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf
11. Croft, W. Bruce and Metzler, Donald and Strohman, Trevor (2015). *Search Engines: Information Retrieval in Practice*. https://ciir.cs.umass.edu/downloads/SEIRiP.pdf
12. Robertson, Stephen and Zaragoza, Hugo (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf
13. Lin, Jimmy and Nogueira, Rodrigo and Yates, Andrew (2021). *Pretrained Transformers for Text Ranking: BERT and Beyond*. https://arxiv.org/abs/2010.06467
14. Wang, Wenhui and Wei, Furu and Dong, Li and Bao, Hangbo and Yang, Nan and Zhou, Ming (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. https://proceedings.neurips.cc/paper/2020/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
15. Yang, An and Yang, Baosong and Zhang, Beichen and Hui, Binyuan and Zheng, Bo and Yu, Bowen and others (2024). *Qwen2.5 Technical Report*. https://arxiv.org/abs/2412.15115
16. Mineault, Patrick (2021). *The Good Research Code Handbook*. https://goodresearch.dev/_static/book.pdf
17. N\"ust, Daniel and Sochat, Vanessa and Marwick, Ben and Eglen, Stephen J. and Head, Tim and Hirst, Tony and Evans, Benjamin D. (2020). *Ten Simple Rules for Writing Dockerfiles for Reproducible Data Science*. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008316
