"""
Evaluation topics, deliberately varied across several axes:
- topic breadth (broad social-science topics vs. narrow technical ones)
- how contested/well-studied the topic is (large conflicting literature
  vs. settled, narrower research bases)
- source availability (mainstream topics have abundant results; niche ones
  stress-test the pipeline when Tavily returns fewer or weaker sources)
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