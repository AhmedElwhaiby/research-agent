"""
~12 evaluation topics, deliberately varied across a few axes rather than
similar-to-each-other:
- topic breadth (broad social-science topics vs. narrow technical ones)
- how contested/well-studied the topic is (lots of conflicting literature
  vs. more settled, narrower research bases)
- how likely search results are to be thin or low-quality (mainstream
  topics have abundant sources; niche ones stress-test what happens when
  Tavily returns fewer/weaker results)

The point isn't to re-confirm what already worked (gaming-and-teens is
already validated manually) — it's to find where the pipeline breaks.
"""

TEST_TOPICS = [
    # broad, well-studied, contested — similar shape to the validated run
    "Effects of remote work on employee productivity",
    "The impact of social media use on adolescent mental health",

    # narrower technical/scientific topics — fewer, denser sources
    "How mRNA vaccines trigger an immune response",
    "The current state of solid-state EV battery technology",

    # niche/less-covered — stress-tests thin search results
    "The economics of vertical farming in urban environments",
    "Preservation challenges for early video game source code",

    # policy/current-events adjacent — sources skew toward news, not papers
    "Global efforts to regulate AI-generated deepfakes",
    "The state of lab-grown meat commercialization",

    # historical/settled — should be easier, good baseline check
    "The causes of the 2008 global financial crisis",
    "How the printing press changed information spread in Europe",

    # deliberately narrow/obscure — plausible worst case for thin sourcing
    "Research on the cognitive effects of bilingualism in older adults",
    "The environmental impact of asteroid mining proposals",
]