        *************************************************************
        *  German version of the English RTE-3 dataset (RTE3-DE)  *
        *************************************************************

                                  March 15, 2013
                          German Research Center for Artificial Intelligence (DFKI GmbH)  
                                  Saarbrücken, Germany


The RTE3-DE dataset is the German translation of the Textual Entailment English dataset used in the RTE-3 Challenge (http://pascallin.ecs.soton.ac.uk/Challenges/RTE3/Datasets/).

Like its English counterpart, the German RTE-3 dataset is composed of a development set and a test set, each containing 800 T/H pairs. RTE3-DE has the following characteristics:
- all T/H pairs were manually translated into German by a native German speaker and proofread by two other native German speakers, all with strong background in CL and NLP
- all information related to the English T/H pairs (e.g. length of T, task)  was imported into the German dataset 
- based on the the judgment of the Italian RTE3 Version, we have identified 8 (5 of the 15 Italian) pairs whose judgment is in disagreement with respect to English set. In the RTE3-DE dataset, the following T/H pairs have a different entailment judgment with respect to the corresponding English ones:
  * DEV SET: IDs 26, 146, 388, 549, 658, 663
  * TEST SET: IDs 663

The RTE3-DE dataset is licensed under a Creative Commons Attribution 3.0 Unported License (http://creativecommons.org/licenses/by/3.0/).

This release contains the following files:

- RTE3-DE-dev-set.xml (800 T/H pairs corresponding to the RTE-3 English development set)
- RTE3-DE-test-set-GS.xml (800 T/H pairs corresponding to the RTE-3 English test set)
- RTE3.dtd (DTD for the dataset)
- RTE3-DE-README.txt (this file)


For further information about this data release, please contact:
  
Günter Neumann	<neumann@dfki.de>

----------------------------------------------------------------------------------------
README created by Günter Neumann on March 15, 2013

Revision: on December 2, 2013 by GN:
Corrected translation of pair 215 in file RTE3-DE-dev-set.xml and changed its decision from NO to YES.
Updated this readme file accordingly.
