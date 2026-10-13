# Genetics reference

Short rules of molecular genetics used to interpret a variant's predicted
consequence. Each section states the rule, when it applies in terms of the
lines the VEP block prints, and its source. The rules name no gene, drug or
patient. Added 2026-10-13 after both models applied these rules from memory,
unstably; here they can be read and quoted.

## reading-frame: Reading frame and coding length

The coding sequence is read in groups of three bases from the start codon.
A deletion or insertion whose coding length is a multiple of three removes
or adds whole codons and leaves the rest of the message read as before; the
protein is shorter or longer but otherwise intact. A length that is not a
multiple of three shifts the grouping, so every codon after the change is
misread until a stop codon is met by chance, and the protein is truncated
with a wrong tail. For deletions in a gene, this distinction separates
frame-preserving losses that leave a partly working protein from
frame-disrupting losses that leave none; the rule was proposed from
patients' deletions and their severity and holds for the large majority.

Applies when: the Frame line gives a count of coding bases deleted or
inserted; check whether the count is a multiple of three. For a deletion that
removes splice sites, use the count for the exons lost whole once the
splicing rule has been applied.

Source: Monaco AP et al., "An explanation for the phenotypic differences
between patients bearing partial deletions of the DMD locus", Genomics 2:90-95
(1988), https://pubmed.ncbi.nlm.nih.gov/3384440/

## splicing: What a transcript keeps of a partly deleted exon

An exon is included in the message only when the splicing machinery
recognises it as an exon: a splice acceptor at its start, a splice donor at
its end, and enough sequence between them for the two sites to be paired
(exon definition). Exons shorter than about 50 bases are hard to define and
the shortest known human exons, a few bases long, are included only with
dedicated regulatory proteins and surrounding signals. A deletion that
removes an exon's splice site and all but a few of its bases leaves a remnant
with no exon definition of its own; under the exon-definition model such a
remnant is not included, and the exon is lost from the message whole. The
same holds for every exon whose splice sites the deletion removes.

Applies when: the block says a deletion covers most but not all of an exon's
coding bases and VEP reports splice acceptor or donor sites among the
consequences. The coding length that then matters is the one for the exons
lost whole, which the block gives beside the count as written.

Source: Berget SM, "Exon recognition in vertebrate splicing", J Biol Chem
270:2411-2414 (1995), https://pubmed.ncbi.nlm.nih.gov/7852296/ ; Ustianenko D
et al., "Microexons: discovery, regulation, and function", WIREs RNA 8:e1418
(2017), https://pubmed.ncbi.nlm.nih.gov/28188674/

## exon-skipping: Restoring the reading frame by skipping a neighbouring exon

When a loss of whole exons is not a multiple of three, removing one more exon
next to the loss can make the total a multiple of three again. An antisense
oligonucleotide that hides that exon from the splicing machinery causes it to
be skipped, so the message joins the exon before it to the exon after the
loss in frame and a shorter protein with intact ends is made. The exon to
skip is therefore the one adjacent to the loss whose length completes a
multiple of three, not an exon inside the loss, which is already absent, and
not a distant exon, which leaves the broken junction in place. Which deletions
are amenable to skipping a given exon follows from this arithmetic.

Applies when: the Frame line shows a whole-exon loss that is not a multiple
of three and the block's neighbouring-exon line names an exon whose removal
gives a multiple of three.

Source: Aartsma-Rus A et al., "Theoretic applicability of antisense-mediated
exon skipping for Duchenne muscular dystrophy mutations", Hum Mutat 30:293-299
(2009), https://pubmed.ncbi.nlm.nih.gov/19156838/

## nonsense-mediated-decay: When a premature stop codon triggers decay of the transcript

A premature stop codon generally triggers nonsense-mediated decay of the
message when it lies more than 50 to 55 bases upstream of the last exon-exon
junction; the message is then degraded and little protein is made. A
premature stop in the last exon, or within the last 50 to 55 bases of the
exon before it, escapes decay, and a truncated protein is made. Further
escapes are described for stops very close to the start codon and for stops
in very long exons.

Applies when: the consequence is stop_gained or frameshift_variant; the block
gives the exon index and the exon count, which say whether the stop lies in
the last exon.

Source: Nagy E, Maquat LE, "A rule for termination-codon position within
intron-containing genes: when nonsense affects RNA abundance", Trends Biochem
Sci 23:198-199 (1998), https://pubmed.ncbi.nlm.nih.gov/9644970/ ; Lindeboom RGH
et al., "The rules and impact of nonsense-mediated mRNA decay in human
cancers", Nat Genet 48:1112-1118 (2016),
https://pubmed.ncbi.nlm.nih.gov/27618451/

## read-through: Suppressing a premature stop codon

A nonsense variant creates a premature stop codon within an otherwise intact
reading frame. Read-through agents let the ribosome insert an amino acid at
that codon and continue, producing full-length protein in a fraction of
translations; efficiency depends on which stop codon it is and its context.
The approach applies to premature stops from substitutions, not to stops
that arise from a frameshift, where reading through the stop would continue
in the wrong frame.

Applies when: the consequence is stop_gained with no change in length.

Source: Keeling KM, Bedwell DM, "Suppression of nonsense mutations as a
therapeutic approach to treat genetic diseases", WIREs RNA 2:837-852 (2011),
https://pubmed.ncbi.nlm.nih.gov/21976286/

## mechanism-direction: Matching a treatment's direction to the variant's effect

A variant can lower a protein's activity (loss of function: a truncating
variant, a destabilising substitution, loss of the gene copy) or raise it
(gain of function: a substitution that keeps the protein active or
prolongs its signal). A drug that blocks or inhibits the protein lowers its
activity, so it suits a gain of function and works against a loss of
function; a drug or supplement that raises the protein's activity, or
supplies what the protein needs to act, suits a loss of function and works
against a gain of function. Knowing which effect the variant has is the
step that orients the choice, and a drug listed as acting on the protein
is not a candidate until its direction matches.

Applies when: the answer is a choice between agents acting on the variant's
protein; the Frame line and consequence (stop_gained, frameshift_variant,
a whole-gene loss) point to a loss of function, while gain of function is
usually a substitution reported as such in the literature for that gene.

Source: Musto E et al., "Recent advances in treatment of epilepsy-related
sodium channelopathies", Eur J Paediatr Neurol 24:123-128 (2020),
https://pubmed.ncbi.nlm.nih.gov/31889633/ ; Brunklaus A et al., "Gene variant
effects across sodium channelopathies predict function and guide precision
therapy", Brain 145:4275-4286 (2022), https://pubmed.ncbi.nlm.nih.gov/35037686/
