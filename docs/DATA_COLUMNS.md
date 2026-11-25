# What are the 287 columns, and why does the model use only 10?

This note is for understanding the data. It explains the raw table that comes out of the Census download, which columns the project uses, and why the rest are left out. Column descriptions come from the Census Bureau's own PUMS data dictionary for 2022 (downloaded with the folktables package), shortened by the Census, so check the official dictionary if you need exact wording.

## 1. Where the 287 columns come from

The project downloads California's 2022 American Community Survey person file (`psam_p06.csv`): **391,171 people, 287 columns**. Each row is one surveyed person. The 287 columns are not 287 pieces of information about a person. Most are bookkeeping.

| Group | Columns | What they are | Used by the model |
|---|---|---|---|
| Identifiers and geography | 8 | Record type, household serial number, region, area codes, income adjustment factor | No |
| Survey weights | 81 | `PWGTP` plus 80 replicate weights (`PWGTP1` to `PWGTP80`), used to scale the sample to the population and to estimate sampling error | No |
| Allocation flags | 74 | One flag per question saying whether the Census had to fill in (impute) that answer | No |
| Survey answers | 124 | Age, education, work, income, disability, migration, language and so on | **10 of them** |
| **Total** | **287** | | |

So only 124 columns are real survey answers, and the model uses 10 of those.

## 2. Why only 10?

The project did not pick these 10. They are the standard definition of the **Folktables ACS Income task** (Ding et al., 2021), the benchmark cited in the survey paper as reference [30]. The code only adapts one name: the Census renamed `RELP` to `RELSHIPP` in 2019, so 2022 data needs the new name.

Reasons for using the standard set:

1. **Most columns cannot be features.** 81 are weights, 74 are allocation flags and 8 are identifiers or geography.
2. **Some columns would give away the answer.** The label is built from total income (`PINCP`), so income columns (wages, self-employment, interest, retirement and so on) would make the model look accurate without learning anything. See section 4.
3. **Comparability.** Published fairness studies use the same 10 columns, so your results can be compared with them.
4. **Readable audits.** Fewer features keep the SHAP importance and the fairness findings easier to explain. Phase 1 tests whether the audit framework works, not how accurate a model can be.

### The 10 columns used

| Column | Meaning (Census wording) | Role |
|---|---|---|
| `AGEP` | Age | feature |
| `COW` | Class of worker | feature |
| `SCHL` | Educational attainment | feature |
| `MAR` | Marital status | feature |
| `OCCP` | Occupation recode for 2018 and later based on 2018 OCC codes | feature |
| `POBP` | Place of birth (Recode) | feature, flagged as a sensitive proxy |
| `RELSHIPP` | Relationship | feature |
| `WKHP` | Usual hours worked per week past 12 months | feature |
| `SEX` | Sex | protected attribute |
| `RAC1P` | Recoded detailed race code | protected attribute |

The label is `PINCP > 50000` (total person income above $50,000), used as a stand-in for "eligible". The task also drops rows, not columns: people aged 16 or under, with income of $100 or less, with zero usual work hours, or with a weight below 1 (391,171 people become 200,577).

## 3. What the cost of this choice is

- The feature set is inherited, not optimised. Other columns could improve accuracy.
- Adding more is not free. Several of them are close stand-ins for national origin, language or disability, which would make the fairness and explainability audits harder to defend (section 5).
- A closer-to-welfare option exists. Folktables also has a public-coverage task (whether someone has public health coverage), which is a welfare-programme outcome and not an income threshold.

## 4. Columns left out on purpose: income (they would leak the label)

| Column | Meaning (Census wording) |
|---|---|
| `PINCP` | Total person's income (use ADJINC to adjust to constant dollars) (this is what the label is made from) |
| `PERNP` | Total person's earnings (use ADJINC to adjust to constant dollars) |
| `WAGP` | Wages or salary income past 12 months (use ADJINC to adjust WAGP to constant dollars) |
| `SEMP` | Self-employment income past 12 months (use ADJINC to adjust SEMP to constant dollars) |
| `INTP` | Interest, dividends, and net rental income past 12 months (use ADJINC to adjust to constant dollars) |
| `OIP` | All other income past 12 months (use ADJINC to adjust to constant dollars) |
| `PAP` | Public assistance income past 12 months (use ADJINC to adjust to constant dollars) |
| `RETP` | Retirement income past 12 months (use ADJINC to adjust to constant dollars) |
| `SSP` | Social Security income past 12 months (use ADJINC to adjust SSP to constant dollars) |
| `SSIP` | Supplementary Security Income past 12 months (use ADJINC to adjust SSIP to constant dollars) |

Note that `PAP` is public assistance income, a welfare payment. It is not in the standard feature set, and it is one of the components of total income, so using it would also leak part of the label.

## 5. Columns left out on purpose: sensitive or proxy columns

These describe race, origin, language or disability. Using them would add fairness risk, and the audit would then have to separate legitimate use from discrimination. `RAC1P` (race), `SEX` and `POBP` (birthplace) are kept deliberately, because the fairness and explainability audits need protected attributes to test against.

| Column | Meaning (Census wording) |
|---|---|
| `RAC2P` | Recoded detailed race code |
| `RAC3P` | Recoded detailed race code |
| `RACAIAN` | American Indian and Alaska Native recode (American Indian and Alaska Native alone or in combination with one or more other races) |
| `RACASN` | Asian recode (Asian alone or in combination with one or more other races) |
| `RACBLK` | Black or African American recode (Black alone or in combination with one or more other races) |
| `RACNH` | Native Hawaiian recode (Native Hawaiian alone or in combination with one or more other races) |
| `RACPI` | Other Pacific Islander recode (Other Pacific Islander alone or in combination with one or more other races) |
| `RACSOR` | Some other race recode (Some other race alone or in combination with one or more other races) |
| `RACWHT` | White recode (White alone or in combination with one or more other races) |
| `HISP` | Recoded detailed Hispanic origin |
| `NATIVITY` | Nativity |
| `CIT` | Citizenship status |
| `CITWP` | Year of naturalization write-in |
| `ANC` | Ancestry recode |
| `ANC1P` | Recoded Detailed Ancestry - first entry |
| `ANC2P` | Recoded Detailed Ancestry - second entry |
| `ENG` | Ability to speak English |
| `LANX` | Language other than English spoken at home |
| `LANP` | Language spoken at home |
| `DDRS` | Self-care difficulty |
| `DEAR` | Hearing difficulty |
| `DEYE` | Vision difficulty |
| `DOUT` | Independent living difficulty |
| `DPHY` | Ambulatory difficulty |
| `DRAT` | Veteran service-connected disability rating (percentage) |
| `DRATX` | Veteran service-connected disability rating (checkbox) |
| `DREM` | Cognitive difficulty |
| `DIS` | Disability recode |

## 6. Appendix A: every survey answer column

All 124 survey answer columns, with what each is and whether it is used.

| Column | Meaning (Census wording) | Status |
|---|---|---|
| `AGEP` | Age | Used: feature |
| `CIT` | Citizenship status | Left out: sensitive or proxy |
| `CITWP` | Year of naturalization write-in | Left out: sensitive or proxy |
| `COW` | Class of worker | Used: feature |
| `DDRS` | Self-care difficulty | Left out: sensitive or proxy |
| `DEAR` | Hearing difficulty | Left out: sensitive or proxy |
| `DEYE` | Vision difficulty | Left out: sensitive or proxy |
| `DOUT` | Independent living difficulty | Left out: sensitive or proxy |
| `DPHY` | Ambulatory difficulty | Left out: sensitive or proxy |
| `DRAT` | Veteran service-connected disability rating (percentage) | Left out: sensitive or proxy |
| `DRATX` | Veteran service-connected disability rating (checkbox) | Left out: sensitive or proxy |
| `DREM` | Cognitive difficulty | Left out: sensitive or proxy |
| `ENG` | Ability to speak English | Left out: sensitive or proxy |
| `FER` | Gave birth to child within the past 12 months | Not used |
| `GCL` | Grandparents living with grandchildren | Not used |
| `GCM` | Length of time responsible for grandchildren | Not used |
| `GCR` | Grandparents responsible for grandchildren | Not used |
| `HIMRKS` | Subsidized Marketplace Coverage | Not used |
| `HINS1` | Insurance through a current or former employer or union | Not used |
| `HINS2` | Insurance purchased directly from an insurance company | Not used |
| `HINS3` | Medicare, for people 65 and older, or people with certain disabilities | Not used |
| `HINS4` | Medicaid, Medical Assistance, or any kind of government-assistance plan for those with low incomes or a disability | Not used |
| `HINS5` | TRICARE or other military health care | Not used |
| `HINS6` | VA (enrolled for VA health care) | Not used |
| `HINS7` | Indian Health Service | Not used |
| `INTP` | Interest, dividends, and net rental income past 12 months (use ADJINC to adjust to constant dollars) | Left out: income, would leak the label |
| `JWMNP` | Travel time to work | Not used |
| `JWRIP` | Vehicle occupancy | Not used |
| `JWTRNS` | Means of transportation to work | Not used |
| `LANX` | Language other than English spoken at home | Left out: sensitive or proxy |
| `MAR` | Marital status | Used: feature |
| `MARHD` | Divorced in the past 12 months | Not used |
| `MARHM` | Married in the past 12 months | Not used |
| `MARHT` | Number of times married | Not used |
| `MARHW` | Widowed in the past 12 months | Not used |
| `MARHYP` | Year last married | Not used |
| `MIG` | Mobility status (lived here 1 year ago) | Not used |
| `MIL` | Military service | Not used |
| `MLPA` | Served September 2001 or later | Not used |
| `MLPB` | Served August 1990 - August 2001 (including Persian Gulf War) | Not used |
| `MLPCD` | Served May 1975 - July 1990 | Not used |
| `MLPE` | Served Vietnam era (August 1964 - April 1975) | Not used |
| `MLPFG` | Served February 1955 - July 1964 | Not used |
| `MLPH` | Served Korean War (July 1950 - January 1955) | Not used |
| `MLPIK` | Peacetime service before July 1950 | Not used |
| `MLPJ` | Served World War II (December 1941 - December 1946) | Not used |
| `NWAB` | Temporary absence from work (UNEDITED - See 'Employment Status Recode' (ESR)) | Not used |
| `NWAV` | Available for work (UNEDITED - See 'Employment Status Recode' (ESR)) | Not used |
| `NWLA` | On layoff from work (UNEDITED - See 'Employment Status Recode' (ESR)) | Not used |
| `NWLK` | Looking for work (UNEDITED - See 'Employment Status Recode' (ESR)) | Not used |
| `NWRE` | Informed of recall (UNEDITED - See 'Employment Status Recode' (ESR)) | Not used |
| `OIP` | All other income past 12 months (use ADJINC to adjust to constant dollars) | Left out: income, would leak the label |
| `PAP` | Public assistance income past 12 months (use ADJINC to adjust to constant dollars) | Left out: income, would leak the label |
| `RELSHIPP` | Relationship | Used: feature |
| `RETP` | Retirement income past 12 months (use ADJINC to adjust to constant dollars) | Left out: income, would leak the label |
| `SCH` | School enrollment | Not used |
| `SCHG` | Grade level attending | Not used |
| `SCHL` | Educational attainment | Used: feature |
| `SEMP` | Self-employment income past 12 months (use ADJINC to adjust SEMP to constant dollars) | Left out: income, would leak the label |
| `SEX` | Sex | Used: protected attribute |
| `SSIP` | Supplementary Security Income past 12 months (use ADJINC to adjust SSIP to constant dollars) | Left out: income, would leak the label |
| `SSP` | Social Security income past 12 months (use ADJINC to adjust SSP to constant dollars) | Left out: income, would leak the label |
| `WAGP` | Wages or salary income past 12 months (use ADJINC to adjust WAGP to constant dollars) | Left out: income, would leak the label |
| `WKHP` | Usual hours worked per week past 12 months | Used: feature |
| `WKL` | When last worked | Not used |
| `WKWN` | Weeks worked during past 12 months | Not used |
| `WRK` | Worked last week | Not used |
| `YOEP` | Year of entry | Not used |
| `ANC` | Ancestry recode | Left out: sensitive or proxy |
| `ANC1P` | Recoded Detailed Ancestry - first entry | Left out: sensitive or proxy |
| `ANC2P` | Recoded Detailed Ancestry - second entry | Left out: sensitive or proxy |
| `DECADE` | Decade of entry | Not used |
| `DIS` | Disability recode | Left out: sensitive or proxy |
| `DRIVESP` | Number of vehicles calculated from JWRI | Not used |
| `ESP` | Employment status of parents | Not used |
| `ESR` | Employment status recode | Not used |
| `FOD1P` | Recoded field of degree - first entry | Not used |
| `FOD2P` | Recoded field of degree - second entry | Not used |
| `HICOV` | Health insurance coverage recode | Not used |
| `HISP` | Recoded detailed Hispanic origin | Left out: sensitive or proxy |
| `INDP` | Industry recode for 2018 and later based on 2017 IND codes | Not used |
| `JWAP` | Time of arrival at work - hour and minute | Not used |
| `JWDP` | Time of departure for work - hour and minute | Not used |
| `LANP` | Language spoken at home | Left out: sensitive or proxy |
| `MIGPUMA` | Migration PUMA based on 2020 Census definition | Not used |
| `MIGSP` | Migration recode - State or foreign country code | Not used |
| `MSP` | Married, spouse present/spouse absent | Not used |
| `NAICSP` | North American Industry Classification System (NAICS) recode for 2018 and later based on 2017 NAICS codes | Not used |
| `NATIVITY` | Nativity | Left out: sensitive or proxy |
| `NOP` | Nativity of parent | Not used |
| `OC` | Own child | Not used |
| `OCCP` | Occupation recode for 2018 and later based on 2018 OCC codes | Used: feature |
| `PAOC` | Presence and age of own children | Not used |
| `PERNP` | Total person's earnings (use ADJINC to adjust to constant dollars) | Left out: income, would leak the label |
| `PINCP` | Total person's income (use ADJINC to adjust to constant dollars) | Source of the label (income above $50,000) |
| `POBP` | Place of birth (Recode) | Used: feature, flagged as a sensitive proxy |
| `POVPIP` | Income-to-poverty ratio recode | Not used |
| `POWPUMA` | Place of work PUMA based on 2020 Census definition | Not used |
| `POWSP` | Place of work - State or foreign country recode | Not used |
| `PRIVCOV` | Private health insurance coverage recode | Not used |
| `PUBCOV` | Public health coverage recode | Not used |
| `QTRBIR` | Quarter of birth | Not used |
| `RAC1P` | Recoded detailed race code | Used: protected attribute |
| `RAC2P` | Recoded detailed race code | Left out: sensitive or proxy |
| `RAC3P` | Recoded detailed race code | Left out: sensitive or proxy |
| `RACAIAN` | American Indian and Alaska Native recode (American Indian and Alaska Native alone or in combination with one or more other races) | Left out: sensitive or proxy |
| `RACASN` | Asian recode (Asian alone or in combination with one or more other races) | Left out: sensitive or proxy |
| `RACBLK` | Black or African American recode (Black alone or in combination with one or more other races) | Left out: sensitive or proxy |
| `RACNH` | Native Hawaiian recode (Native Hawaiian alone or in combination with one or more other races) | Left out: sensitive or proxy |
| `RACNUM` | Number of major race groups represented | Not used |
| `RACPI` | Other Pacific Islander recode (Other Pacific Islander alone or in combination with one or more other races) | Left out: sensitive or proxy |
| `RACSOR` | Some other race recode (Some other race alone or in combination with one or more other races) | Left out: sensitive or proxy |
| `RACWHT` | White recode (White alone or in combination with one or more other races) | Left out: sensitive or proxy |
| `RC` | Related child | Not used |
| `SCIENGP` | Field of Degree Science and Engineering Flag - NSF Definition | Not used |
| `SCIENGRLP` | Field of Degree Science and Engineering Related Flag - NSF Definition | Not used |
| `SFN` | Subfamily number | Not used |
| `SFR` | Subfamily relationship | Not used |
| `SOCP` | Standard Occupational Classification (SOC) codes for 2018 and later based on 2018 SOC codes | Not used |
| `VPS` | Veteran period of service | Not used |
| `WAOB` | World area of birth | Not used |
| `FHINS3C` | Medicare coverage given through the eligibility coverage edit | Not used |
| `FHINS4C` | Medicaid coverage given through the eligibility coverage edit | Not used |
| `FHINS5C` | TRICARE coverage given through the eligibility coverage edit | Not used |

## 7. Appendix B: the other columns, by name

**Identifiers and geography (8):** `RT` (Record Type), `SERIALNO` (Housing unit/GQ person serial number), `DIVISION` (Division code based on 2010 Census definitions Division code based on 2020 Census definitions), `SPORDER` (Person number), `PUMA` (Public use microdata area code (PUMA) based on 2020 Census definition (areas with population of 100,000 or more, use with ST for unique code)), `REGION` (Region code based on 2020 Census definitions), `ST` (State Code based on 2020 Census definitions), `ADJINC` (Adjustment factor for income and earnings dollar amounts (6 implied decimal places)).

**Survey weights (81):** `PWGTP` and `PWGTP1` to `PWGTP80`.

**Allocation flags (74):** `FAGEP`, `FANCP`, `FCITP`, `FCITWP`, `FCOWP`, `FDDRSP`, `FDEARP`, `FDEYEP`, `FDISP`, `FDOUTP`, `FDPHYP`, `FDRATP`, `FDRATXP`, `FDREMP`, `FENGP`, `FESRP`, `FFERP`, `FFODP`, `FGCLP`, `FGCMP`, `FGCRP`, `FHICOVP`, `FHIMRKSP`, `FHINS1P`, `FHINS2P`, `FHINS3P`, `FHINS4P`, `FHINS5P`, `FHINS6P`, `FHINS7P`, `FHISP`, `FINDP`, `FINTP`, `FJWDP`, `FJWMNP`, `FJWRIP`, `FJWTRNSP`, `FLANP`, `FLANXP`, `FMARP`, `FMARHDP`, `FMARHMP`, `FMARHTP`, `FMARHWP`, `FMARHYP`, `FMIGP`, `FMIGSP`, `FMILPP`, `FMILSP`, `FOCCP`, `FOIP`, `FPAP`, `FPERNP`, `FPINCP`, `FPOBP`, `FPOWSP`, `FPRIVCOVP`, `FPUBCOVP`, `FRACP`, `FRELSHIPP`, `FRETP`, `FSCHGP`, `FSCHLP`, `FSCHP`, `FSEMP`, `FSEXP`, `FSSIP`, `FSSP`, `FWAGP`, `FWKHP`, `FWKLP`, `FWKWNP`, `FWRKP`, `FYOEP`.



Set-Location E:\MTECH\Final_yr_project; git add docs/DATA_COLUMNS.md; git commit -m "Add data column guide"; git push