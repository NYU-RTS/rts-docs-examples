
# README — Scanned documents to data base creation (Phase 1)

This repository contains the full workflow for converting scanned documents into clean, readable, structured text files using OCR and LLM-based post-processing.

The workflow is implemented in a Jupyter Notebook (`Workflow.ipynb`).

---

# Project Overview

The goal of this project is to take scanned archival newsletters from the *Women Artists Newsletter / Women Artists News* collection and convert them into:

1. OCR text files
2. Clean, human-readable text
3. Organized outputs suitable for indexing, search, metadata extraction, RAG systems, and mainly, to create a database of this information.

This README explains the entire workflow used in the notebook and supporting scripts.

---

# Repository Structure

```
.
├── Newsletter_OCR_LLM_Project/
│   ├── OCR_versions/               # Raw OCR .txt files
│   ├── parsable_versions/          # Cleaned text files (LLM formatted)
├── Workflow.ipynb                  # Full end-to-end pipeline
├── README.md
```

---

# 1. Data Source (Original PDFs)

![Article from the newsletter](Newsletter_OCR_LLM_Project/images/article.png)
![Calendar Exhibits/Group Shows section](Newsletter_OCR_LLM_Project/images/calendar_exhibits.png)
![First Page of the Newsletter](Newsletter_OCR_LLM_Project/images/first_page.png)

We begin with scanned issues of the *Women Artists Newsletter*. As we can see, these scans contain irregular formatting, artifacts, shadows, multi-column layouts, and inconsistent spacing, making them unsuitable for direct text analysis.

Many different methods was experimented with for text extraction, including some vision language models, however, they all conatained inconsistencies. The next step was to try LLMs directly. After much experimentation with Claude models, we found that certain errors kept reoccuring and providing examples overfit the model to that error, but not all. 

For eg: "Susan Weitzman" always showed up as "Susan Wolfsman" or some other iteration of this.

In this use case, it was incredibly important to have the *correct* entity names. The LLM relied on it's own historic knowledge and hallucinated often, especially when text was in fine print or small in font size, "u" would show up as "o", or "c" would should up as an "r". 

After much research and experimentation, using a pre trained OCR model such as Amazon textract, or Google vision seemed like the appropriate next step. Google Vision showed extremely promising results. This leads us to the OCR phase.

The entire process is available as a hands on in the Workflow.ipynb file.

---

# 2. Generating OCR Output

OCR (performed via GCP) produces `.txt` files stored in:

```
Newsletter_OCR_LLM_Project/OCR_versions/
```

Example filenames:

```
1976_06-01_Vol.2_No.3_compressed_ocr.txt

```

- In this section, we loop through all the files in our data folder, to extract any and all information in the PDF we have. We use Google Cloud (GCP) for this. However, you can switch to any other cloud OCR model of your choice. You will have to create your own GCP project, and use your credentials (GCP provides $300 free credits for all new  users, for 3 momths)
- To note:
     -  In this project, the professor had no use case for the first page of every pdf, so the code explicitly skips the first page (0 in python index)

 Here's an example of what the output looks like:

 """
 --- Newsletter Page 1 (PDF page 2) ---
photo Gina Shamus
photo: Carole Rosen
ISSN 0149 7081
75 cents
Women Artists News
Vol. 4 No. 9
First Annual WCA Awards for Outstanding Achievement in the Visual Arts
photo: Carole Rosen
Joan Mondale and Mary Ann Tighe
join the applause for Louise Nevelson
On the White House lawn, after presentation of awards by President in Oval Office:
(from left) Charlotte Robinson, Selma Burke, Louise Nevelson, Alice Neel, Ann
Sutherland Harris, Isabel Bishop, Lee Anne Miller
Sen. Harrison Williams addresses the Coali-
tion of Women's Art Organizations.; Joyce
Aiken, Judith Brodsky, and Louise Wiener
at the dais (see page 2 for Coalition story)
GHON A
CAJOUS FOR HET
photo: Carole Rosen
photo: Gina Shamus
THE
EMBASSY RO
WOMEN
SPEAK TO
Awards Ceremony
Isabel Bishop, Selma Burke, Alice Neel, Lou-
ise Nevelson, and Georgia O'Keeffe received
citations from President Carter in a ceremo-
ny at the White House Jan. 30. They were
then honored by the WCA at the Embassy
Row Hotel in the First Annual WCA Out-
standing Achievement in the Visual Arts
Awards ceremony.
Speakers at the WCA ceremony were:
1979, Women Artists News
"""

#### As we can see, the output here is all in a single column, and has certain errors in heirarchy. 

The raw OCR text typically contains:

* broken line breaks
* merged or split lines
* inconsistent spacing
* hyphenated words
* noise from headers, footers, and page numbers

This output must be cleaned before use, we need to make sure that any information that is extracted or parsed, has the correct creditentials for each article. To fix this, we use an LLM. This leads to the next phase : *The LLM stage*


---

# 3. LLM-Based Post-Processing (Cleaning and Formatting to create Parsable version of the text)

To transform the OCR text into human-readable form, the workflow uses the Claude model via Portkey (NYU AI Gateway).

The model performs:

* Meta data extraction 
    - This includes article type, date, volume, issue, authors/contributors 
* It then puts the actual content in clear paragraph breaks that is easily readable to a human being. 

### Note: We instruct the LLM to keep things verbatim, this makes sure that there is no hallucination. If we let the LLM "correct minor errors", it may hallucinate and fix entity names that are actually correct. The cleaned therefore remains faithful to the original and avoids hallucination.

This process is conducted by defining all our rules in a clearly structure prompt. This prompt can be found in the Wokrflow.ipynb file, or in the separate prompts.txt file

---

# 4. Prompt Design

The prompt we used follows the following structure:

* Defines the model’s role as an **archivist** restructuring raw OCR text without altering content.
* States the **goal**: produce a clean, readable plain-text version while preserving every word verbatim.
* Provides **rules for restructuring**, including merging split articles, fixing bylines, preserving lists, and converting column text into paragraphs.
* Requires **categorizing each section** (article, calendar, masthead, advertisement, etc.) and outputting them in a specific order.
* Specifies a strict **section template** (TITLE, WRITER, PAGE_NUMBERS, VOLUME, ISSUE, SEASON_YEAR, TYPE, CONTENT).
* Includes an **example** demonstrating the exact expected output format.

---

# 5 Single-File & Batch Processing

* In the Workflow.ipynb file, you will find the first code allows you process a single file. This helps to experiment with prompt outputs and make any required changes. 
* The second code allows you to run all the files in loop, i.e as a batch. Processing them together can make the process much faster. However, make sure your final prompt can be generalised to all task cases. Incase different data/files have different structures or different ways of presenting content, the prompt may fail to give good results all the time. 

---

# 6. Final Output Example: 

TITLE: GENDER IN ART AN ONGOING DIALOGUE
WRITER: Sophie Rivera
PAGE_NUMBERS: 1, 4
VOLUME: 1
ISSUE: 8
SEASON_YEAR: January 1976
TYPE: PANEL
CONTENT:
A group of women artists decided to express their growing disenchantment with the women's movement. "Gender in Art An Ongoing Dialogue," was a title just vague enough to attract a large turnout of artists at the AIR gallery. Following a brief slide presentation, moderator/artist Nancy Spero set the tone with an opening statement full of vague references to a meeting the artists had held several years ago. Spero mentioned neither the topics discussed nor the conclusions formulated, just that they had discussed "a common bond between women."

The panel, in trying to redefine "feminist," "feminine," and "female," were unable to agree, but initially opted for "female." According to artist Rosemary Mayer "a feminist esthetic is a very precise thing; a feminine esthetic is a lousy term; and a female esthetic could possibly have meaning." Before Mayer could elaborate on the "possibility of meaning," artist/anthropologist Elizabeth Weatherford challenged the choice of "female." She preferred "feminist" to describe women artists' work but conceded that "certain stylistic choices are made."

Critic Lucy Lippard said, "If a woman is thinking about her work as by a woman she is probably pre-feminist, post-feminist, or something-or-other-feminist." Artist Nancy Kitchel said, "so little imagery is left to be applied to female, feminine, and feminist art." After using the panel's terms, Kitchel bemoaned the fact that "art has been separated by its terminology out of the stream of human activity" so far as to become a "separate category alien to the artists' intentions."

Spero pointed out that Rosemary Mayer's sculptures were titled with the names of great and powerful women. Yet Mayer claimed her intention was not really feminist. "My work was feminist to the extent that I thought people should be aware of the lives and activity of those women. It was not feminist to the extent that I thought those forms were female," answered Mayer. She elaborated on the stereotypes associated with art done with stitching and fabric. There was no general agreement about the relevance of techniques learned by women growing up and their application to a feminist consciousness in art.

The discussion had little to do with the stated subject. Some of the panelists commented on the male dominance of the art world--a theme which surfaced early, got lost, then re-surfaced in response to sharp audience questioning. The audience expressed feelings of powerlessness in a male dominated society. Artist Joan Semmel answered that women are our audience, that women have a gut response to art, and that her own art came out of a sense of powerlessness (although Semmel no longer feels powerless). Spero strongly disagreed. One could not help get the feeling that we were listening to an economic theory, that many of the women were interested only in the marketing and marketability of feminist art.

The heart of the dilemma seems to be the intrinsic value versus the extrinsic commodity value of art. As to whether there is a specific female art form--a panelist asserted that the traditional female approach has been to reach out, while the male approach has been to look into himself in order to create. This was directly contradicted by statements of at least half a dozen women about their own creativity.

The confusion deepened when someone mentioned that she had been reading a book claiming that people were pushed, because of education, away from the visual toward the verbal. This led to the speculation that female and male spatial perceptions are different--a useful statement, if true, but taken wholly out of context.

The discussion might more appropriately have been titled: "Disgruntled Artists Lower Consciousness." Despite claims of innovation, the ground had been gone over before.

TITLE: THOUGHTS PROVOKED BY A "GENDER-IN-ART" PANEL
WRITER: Joan Semmel
PAGE_NUMBERS: 1, 4
VOLUME: 1
ISSUE: 8
SEASON_YEAR: January 1976
TYPE: ARTICLE
CONTENT:
The impetus for the woman's movement in the art world was blatant discrimination, exclusion and isolation. It was important for many of us in the early years to have the opportunity to see each other's work, and to gain the confidence to further develop and expand our own work. We then returned to our private worlds to work intensively, gradually gaining exposure, first in women's shows, then in wider contexts.

The profusion of women's panels this season is a signal that we are once again seeking nurturance from each other, and that the movement is readying itself for the next stage in its development. Unfortunately many of the panels have failed to deal with the substantive issues and have left us with an aftertaste of frustration and negation.

A panel that calls itself "Gender In Art-An Ongoing Dialogue" and then refuses to deal with content or sources, or gender itself except in terms of careerism, does a disservice to us all.

Because women's work has been discriminated against for years, many women are paranoid about having their art described as distinctively female, feminist, or feminine. Some think women's art should be accepted because it is the same, or as good as, men's. I want it to be accepted because it is different. Therein lie its power and its possibilities.

....

---

# 10. Bonus Use Case : Indexing

* Using the prompt for creating a detailed index, you can generate a detailed list highlighting all contubutors, articles, entities advertisements etc in the file. The final outputs can be found in the index_result folder.

# 9. Notebook Workflow Summary (Workflow.ipynb)

The notebook performs:

* loading OCR files
* cleaning text with the LLM
* inspecting excerpts
* saving outputs
* creating an index

The notebook serves as the master workflow that integrates the single-file and batch codes.

---

# 10. Requirements

The workflow.ipynb requires:

* All libraires mentioned in the requirements.txt file
* Portkey credentials (contact your PI, or, Research Technology Services (RTS) at NYU for access)
* Google Cloup Platform (GCP) credentials, $300 available in free credit for each new user, or, contact RTS.

___

# 11. Final outputs:

* This method can be applicable for unstructured data such as this newsletter. 
* The final output of this project, scaled to about 100 files, and about 2000 pages, was used to create a data base of all artists and contributors mentioned in the Women Arts Newsletter.
* These results can also by easily used for creating a RAG workflow, where the underlying data is all the structured extracted/generated materials.