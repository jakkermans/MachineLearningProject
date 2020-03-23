------------------------------------------------------------------------------------------
CLiPS Stylometry Investigation (CSI) Corpus
------------------------------------------------------------------------------------------

Creator(s):
	CLiPS Research Center, University of Antwerp
	Ben Verhoeven, Walter Daelemans
	www.clips.uantwerpen.be

Version: 2015-10
Language: Dutch

This dataset is available at http://www.clips.uantwerpen.be/datasets
Previous versions of the dataset remain available from the authors via e-mail request.

License:
	The dataset is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 
	3.0 Unported License. Please read the terms of use carefully.
	http://creativecommons.org/licenses/by-nc-sa/3.0
	(Full legal code enclosed: License.txt)
	For other uses, please contact Walter Daelemans (walter.daelemans@uantwerpen.be)

Description:
	The CSI corpus is a yearly expanded corpus of student texts in two genres: essays and 
	reviews. The purpose of this corpus lies primarily in stylometric research, but many other 
	applications are possible. There is a vast amount of meta-data available, both on the 
	author (gender, age, sexual orientation, region of origin, personality profile) and 
	on the document (timestamp, genre, topic, veracity, sentiment).
	
If you use this dataset in your research, make sure to cite the following paper:

	Verhoeven, Ben & Daelemans, Walter (2014). CLiPS Stylometry Investigation (CSI) 
	corpus: A Dutch corpus for the detection of age, gender, personality, sentiment and 
	deception in text. In: Proceedings of the 9th International Conference on Language 
	Resources and Evaluation (LREC 2014). Reykjavik, Iceland.
	(Full paper at http://www.clips.uantwerpen.be/bibliography/clips-stylometry-investigation-csi-corpus)

Acknowledgement:
	We would like to express our gratitude to Katrien Verreyken, Shanti Verellen, Sarah 
	Van Hoof, Dominiek Sandra and Reinhild Vandekerckhove (University of Antwerp) for 
	their help in collecting all the data.
	This corpus was first constructed within the framework of the AMiCA project, funded 
	by the Flemish Agency for Innovation through Science and Technology (IWT), but its 
	further development is supported by a PhD grant of FWO - Research Foundation Flanders 
	for the first author.


------------------------------------------------------------------------------------------
Corpus Structure
------------------------------------------------------------------------------------------

The corpus is divided into two folders: reviews & essays, one for each genre.
There is a third folder, named 'reviews_propname', which contains the same reviews as in 
the original data, but all the product names have been replaced with a *propname* tag. 
These reviews can be used for deception or topic detection where there is a one-to-one 
relationship between the class and the product names.

The texts included in this corpus are formatted in UTF-8 txt-files.

The text files are named as follows:
- for essays/papers  -> AuthorID_Genre_Timestamp.txt (*)
- for reviews -> AuthorID_Genre_Timestamp_Veracity_Sentiment.txt

(*) When more than one essay/paper exists of the same author and date, '+' is added to
	the file name.
	
There are two lists of metadata available, one with data about the authors, another one
with data about the documents.
- List.CSI.AuthorData.1.3.1.BV.2015-05-11.txt
- List.CSI.DocumentData.1.3.2.BV.2015-10-01.txt

The metadata files contain tab-separated values with each line representing one author 
or document. They are structured as follows:

AUTHOR DATA
AuthorID DateOfBirth Gender SexualPreference Region Country BigFive MBTI

DOCUMENT DATA
FileName AuthorID Timestamp Genre Grade Sentiment Veracity Category Product Subject

AuthorID = 8-digit unique id for the author
DateOfBirth = birth date of author, structured as yyyy-dd-mm (or yyyy)
Gender = gender of author
SexualPreference = sexual preference of author (straight or gay/LGBT)
Region = region where author grew up (Belgian province / The Netherlands or Other)
Country = country where author grew up
BigFive = Big Five personality scores of author, separated by '-'
	order of traits: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticity (= OCEAN)
MBTI = Myers-Briggs personality scores of author, separated by '-'

FileName = file name of this document
Timestamp = writing date of the text, structured as yyyy-dd-mm (or yyyy)
Genre = genre of the document
Grade = grade (out of 20) given by professor to the document 
Sentiment = sentiment of the review, either positive or negative
Veracity = indicates whether a text is truthful or deceptive
Category = the topic of the review (e.g. smartphones, books, restaurants, ...)
Subject = the product under review (e.g. iPhone 5, Lord of the Rings, ...)

When a personality score is not present, its place is left blank, but the '-' marks remain.
When certain information is not applicable or not available, we used 'NA' instead.


------------------------------------------------------------------------------------------
Corpus Statistics
------------------------------------------------------------------------------------------

Table 1. Document statistics per genre

Genres		#docs	#tokens		Avg. length		Std.dev.
---------------------------------------------------------
Reviews		1298	202,827		156				65
Essays		517		565,885		1095			734
---------------------------------------------------------
Total		1815	768,712		


Table 2. Distribution of reviews over types

			Positive	Negative	Total
---------------------------------------------------------
Truth		323			326			649
Deception	319			330			649
---------------------------------------------------------
Total		642			656			1298


Table 3. Distribution of author gender and sexual orientation

Gender	Straight	LGBT(*)		Unknown		Total
---------------------------------------------------------
Male	92			10			32			134
Female	368			13			146			527
---------------------------------------------------------
Total	460			23			178			661

(*) LGBT = Lesbian, Gay, Bisexual or Transgender


Table 4. Distribution of origin of author (Dutch-speaking Belgian provinces, 
The Netherlands or other)

Region				# authors	% authors
---------------------------------------------
Antwerpen			434			65.7
Limburg				38			5.8
Vlaams-Brabant		24			3.6
Oost-Vlaanderen		65			9.8
West-Vlaanderen		26			3.9
The Netherlands		64			9.7
Other				10			1.5
---------------------------------------------
Total				661			100


Table 5. Average Big Five personality profile of the authors in the corpus

Openness	Conscientiousness	Extraversion	Agreeableness	Neuroticity
  50.0			46.3				53.0			43.0			53.7
  
We have a Big Five personality profile available for 535 of the authors.
We also have an MBTI personality profile for 435 of the latter group.

