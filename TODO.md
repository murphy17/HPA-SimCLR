# TODO

- Finish refactoring
    - The results figures(?) - 7_Additional_Figures?
    - ***enrichemnt analysis***

- Run COMET
    - ***figure out how to produce ROC!!!***

- Testis
    - Train on testis
    - Download ghosal predicitons

- Final fomratting
    - Regenerate figures @ 300dpi ***including the UMAP***
    - Figures ordering
    - a) Your revised manuscript, in PDF format, HIGHLIGHTING changes in RED font.
    - b) A letter, in PDF format, describing how you have addressed the reviews
    - c) A zip archive containing all your source files, INCLUDING:
     - The manuscript as a Word or Latex file *using the template* provided on the Bioinformatics web site
     - Each figure in a separate file at publication quality resolution, either as vector graphics (PDF, EPS) 
       or as high-resolution .tif files (1200 dpi for line drawings and 300 dpi for color and half-tone artwork)
     - Supplementary material if applicable
     - A clean PDF version of your manuscript
    
- Rebuttal letter

- Writing
    - ***Table explaining datasets***
    - ***Donor analysis***
    - ROCs for ablated model
    - Supplement: testis analysis, procedure (Scanpy, how I labelled)
    - Added t-test
    - Removal of Platt scaling
    - replace MI discussion with Classification accuracy of linear model
    - Weasel out of antibodies
    - Justify Densenet
    - There's really nothing out there like HPA, although this approach can be applied once trained to any IHC image

- Supplement figure on kidney cancer images?

# DONE

- Run no_importance, no_grouping

- Test run the example
    - hceckpoint loading odesnt work
    - Write the readme
    
- throw in t-test as well

- Generate ROCs for ablated model as suggested
    - classification performance is basically the same.
    - one possibility is that we reduced the effective sample size by doing this
    - another is that the relevant invariances are already baked into imagenet to an extent
    - HOWEVER... show that you at least can't predict the donor as well...
    - put in supplement