# Examples (k = 5)

## coverage_mod completes, top-k does not (6 questions)

- **qid 173** (asdaaburson-marstellerarabyouthsurvey2014-140407100615-phpapp01_95.pdf, Chart): In the year in which Palestine was added to the survey, respondents who believe traditional values are outdated increased by how many percentage points compared to 2011?
  - answer: 29% | gold pages [6, 10]
  - needs: ['year Palestine was added to the survey', 'percentage of respondents who believe traditional values are outdated in the year Palestine was added to the survey', 'percentage of respondents who believe traditional values are outdated in 2011']
  - topk: pages [10, 11, 1, 45, 15] -> 39
  - coverage: pages [11, 1, 10, 45, 15] -> 39
  - coverage_mod: pages [11, 10, 6, 1, 45] -> 30
  - gold: pages [6, 10] -> 39
- **qid 348** (PP_2019.01.17_Trump-economy_FINAL2.pdf, Chart): In the year when 58% of people thought it was too early to tell if Trump was a successful president, how many people believed that his economic policies had not much effect on the economic situation?
  - answer: 49% | gold pages [13, 14]
  - needs: ['year when 58% of people thought it was too early to tell if Trump was a successful president', "percentage of people who believed Trump's economic policies had not much effect on the economic situation"]
  - topk: pages [13, 4, 5, 3, 7] -> 64
  - coverage: pages [13, 14, 4, 5, 3] -> 64
  - coverage_mod: pages [4, 14, 13, 5, 3] -> 64
  - gold: pages [13, 14] -> 26
- **qid 457** (STEPBACK.pdf, Table): How much lower is the accuracy of GPT-4 on SituatedQA compared to MMLU Chemistry?
  - answer: 16.7% | gold pages [5, 6]
  - needs: ['accuracy of GPT-4 on SituatedQA', 'accuracy of GPT-4 on MMLU Chemistry']
  - topk: pages [1, 4, 6, 7, 13] -> 20.8
  - coverage: pages [1, 5, 4, 6, 7] -> 28.7
  - coverage_mod: pages [1, 5, 4, 6, 7] -> 28.7
  - gold: pages [5, 6] -> 18.7
- **qid 570** (PS_2018.01.09_STEM_FINAL.pdf, Chart + Table + Text): A particular racial group was treated least fairly in opportunities for promotion and advancement, as well as in recruitment and hiring processes in the survey of Pew Research Center, January 2018. What is the percentage of this group been denied a promotion in STEM jobs?
  - answer: 14% | gold pages [78, 82]
  - needs: ['racial group treated least fairly in opportunities for promotion and advancement in Pew Research Center survey January 2018', 'percentage of the racial group denied a promotion in STEM jobs']
  - topk: pages [82, 14, 83, 8, 15] -> 25
  - coverage: pages [82, 14, 78, 83, 8] -> 14
  - coverage_mod: pages [82, 14, 78, 83, 8] -> 14
  - gold: pages [78, 82] -> 14

## top-k completes, coverage_mod does not (1 questions)

- **qid 503** (PRE_2022.09.29_NSL-politics_REPORT.pdf, Chart + Layout): Which Hispanic origin group in the United States is most likely to vote for Republican candidate for the U.S. House of Representatives in their district and to say Trump should run for president in 2024?
  - answer: Puerto Rican | gold pages [57, 66]
  - needs: ['Hispanic origin group in the United States most likely to vote for Republican candidate for the U.S. House of Representatives', 'Hispanic origin group in the United States most likely to say Trump should run for president in 2024']
  - topk: pages [57, 56, 14, 17, 66] -> Cuban
  - coverage: pages [17, 14, 65, 57, 56] -> Cuban
  - coverage_mod: pages [17, 14, 65, 57, 56] -> Cuban
  - gold: pages [57, 66] -> Cuban

## top-k complete but wrong, gold also wrong (reader failure) (92 questions)

- **qid 1** (PH_2016.06.08_Economy-Final.pdf, Table): According to the report, which one is greater in population in the survey? Foreign born Latinos, or the Latinos interviewed by cellphone?
  - answer: Latinos interviewed by cellphone | gold pages [19, 20]
  - needs: ['population of Foreign born Latinos in the survey', 'population of Latinos interviewed by cellphone in the survey']
  - topk: pages [19, 20, 5, 21, 10] -> Foreign born Latinos
  - coverage: pages [19, 20, 5, 21, 10] -> Foreign born Latinos
  - coverage_mod: pages [19, 20, 5, 21, 10] -> Foreign born Latinos
  - gold: pages [19, 20] -> Foreign born Latinos
- **qid 3** (PH_2016.06.08_Economy-Final.pdf, Text): Among all 12 references in this report, how many are from its own research center?
  - answer: 8 | gold pages [2, 22, 23]
  - needs: ["references from the report's own research center"]
  - topk: pages [22, 23, 3, 2, 18] -> 6
  - coverage: pages [18, 22, 23, 3, 2] -> 6
  - coverage_mod: pages [18, 22, 23, 3, 2] -> 6
  - gold: pages [2, 22, 23] -> 7
- **qid 7** (Independents-Report.pdf, Text): What's the percentage of people who are democrats and voted in the last election compared to the entire population in 2018?
  - answer: 18.29% | gold pages [3, 5]
  - needs: ['percentage of democrats who voted in the last election in 2018', 'entire population in 2018']
  - topk: pages [4, 5, 3, 22, 20] -> 61
  - coverage: pages [3, 4, 5, 22, 20] -> 61
  - coverage_mod: pages [3, 4, 22, 5, 20] -> 61
  - gold: pages [3, 5] -> 31
- **qid 57** (reportq32015-151009093138-lva1-app6891_95.pdf, Figure + Layout): In Q3 2015, what is the approximate range of cost in 1 day (dollar) of a mobile incentive platform in Vietnam for installing? Answer with a list of two number for boundary, for example [3000, 4000]
  - answer: [1500, 8000] | gold pages [26, 27]
  - needs: ['cost of mobile incentive platform in Vietnam for installing in Q3 2015', 'range of cost in 1 day (dollar) of mobile incentive platform in Vietnam for installing in Q3 2015']
  - topk: pages [26, 27, 32, 13, 21] -> ['0.3', '0.4']
  - coverage: pages [27, 26, 32, 13, 21] -> ['0.3', '0.4']
  - coverage_mod: pages [11, 26, 27, 32, 22] -> ['0.3', '0.4']
  - gold: pages [26, 27] -> ['1500', '2000']
