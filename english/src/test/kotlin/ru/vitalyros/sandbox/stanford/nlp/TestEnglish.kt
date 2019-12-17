package ru.vitalyros.sandbox.stanford.nlp

import org.junit.Test
import edu.stanford.nlp.pipeline.CoreDocument
import edu.stanford.nlp.pipeline.StanfordCoreNLP
import java.util.*


class TestEnglish {
    val en_text = "Joe Smith was born in California. " +
            "In 2017, he went to Paris, France in the summer. " +
            "His flight left at 3:00pm on July 10th, 2017. " +
            "After eating some escargot for the first time, Joe said, \"That was delicious!\" " +
            "He sent a postcard to his sister Jane Smith. " +
            "After hearing about Joe's trip, Jane decided she might go to France one day."

    @Test
    fun test_split_en() {
        val props = Properties()
        props.setProperty("annotators", "tokenize,ssplit")
        val pipeline = StanfordCoreNLP(props)
        val document = CoreDocument(en_text)
        pipeline.annotate(document)
        document.sentences().forEach { sentence ->
            println("sentence: ${sentence.text()}")
            println("tokens: ${sentence.tokens().map { "${it.originalText()} ${it.index()}"}.joinToString(", ", "[", "]")}")
        }
    }

    /**
    1. 	    CC 	Coordinating conjunction
    2. 	    CD 	Cardinal number
    3. 	    DT 	Determiner
    4. 	    EX 	Existential there
    5. 	    FW 	Foreign word
    6. 	    IN 	Preposition or subordinating conjunction
    7. 	    JJ 	Adjective
    8. 	    JJR 	Adjective, comparative
    9. 	    JJS 	Adjective, superlative
    10. 	LS 	List item marker
    11. 	MD 	Modal
    12. 	NN 	Noun, singular or mass
    13. 	NNS 	Noun, plural
    14. 	NNP 	Proper noun, singular
    15. 	NNPS 	Proper noun, plural
    16. 	PDT 	Predeterminer
    17. 	POS 	Possessive ending
    18. 	PRP 	Personal pronoun
    19. 	PRP$ 	Possessive pronoun
    20. 	RB 	Adverb
    21. 	RBR 	Adverb, comparative
    22. 	RBS 	Adverb, superlative
    23. 	RP 	Particle
    24. 	SYM 	Symbol
    25. 	TO 	to
    26. 	UH 	Interjection
    27. 	VB 	Verb, base form
    28. 	VBD 	Verb, past tense
    29. 	VBG 	Verb, gerund or present participle
    30. 	VBN 	Verb, past participle
    31. 	VBP 	Verb, non-3rd person singular present
    32. 	VBZ 	Verb, 3rd person singular present
    33. 	WDT 	Wh-determiner
    34. 	WP 	Wh-pronoun
    35. 	WP$ 	Possessive wh-pronoun
    36. 	WRB 	Wh-adverb
     */
    @Test
    fun test_split_pos_en() {
        val props = Properties()
        props.setProperty("annotators", "tokenize,ssplit,pos")
        val pipeline = StanfordCoreNLP(props)
        val document = CoreDocument(en_text)
        pipeline.annotate(document)
        document.sentences().forEach { sentence ->
            println("sentence: ${sentence.text()}")
            println("tokens: ${sentence.tokens().map { "${it.originalText()} ${it.index()} ${it.tag()}"}.joinToString(", ", "[", "]")}")
            println("pos tags: ${sentence.posTags()}")
            println()
        }
    }

    @Test
    fun test_split_lemma_en() {
        val props = Properties()
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma")
        val pipeline = StanfordCoreNLP(props)
        val document = CoreDocument(en_text)
        pipeline.annotate(document)
        document.sentences().forEach { sentence ->
            println("sentence: ${sentence.text()}")
            println("tokens: ${sentence.tokens().map { "${it.originalText()} ${it.index()} ${it.lemma()}"}.joinToString(", ", "[", "]")}")
        }
    }

    @Test
    fun test_split_ner_en() {
        val props = Properties()
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner")
        val pipeline = StanfordCoreNLP(props)
        val document = CoreDocument(en_text)
        pipeline.annotate(document)
        document.sentences().forEach { sentence ->
            println("sentence: ${sentence.text()}")
            println("tokens: ${sentence.tokens().map { "${it.originalText()} ${it.index()} ${it.ner()} ${it.nerConfidence()}"}.joinToString(", ", "[", "]")}")
            println("nre tags: ${sentence.nerTags()}")
            println()
        }
    }


    /**
     * https://en.wikipedia.org/wiki/Phrase_structure_grammar
     * https://en.wikipedia.org/wiki/Treebank
     */
    @Test
    fun test_split_constituency_parse_en() {
        val props = Properties()
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse")
        val pipeline = StanfordCoreNLP(props)
        val document = CoreDocument(en_text)
        pipeline.annotate(document)
        document.sentences().forEach { sentence ->
            println("sentence: ${sentence.text()}")
            println("tokens: ${sentence.tokens().map { "${it.originalText()} ${it.index()}"}.joinToString(", ", "[", "]")}")
            println("constituency parse: ${sentence.constituencyParse()}")
            println()
        }
    }


    @Test
    fun test_split_dependency_parse_en() {
        val props = Properties()
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,depparse")
        val pipeline = StanfordCoreNLP(props)
        val document = CoreDocument(en_text)
        pipeline.annotate(document)
        document.sentences().forEach { sentence ->
            println("sentence: ${sentence.text()}")
            println("tokens: ${sentence.tokens().map { "${it.originalText()} ${it.index()}"}.joinToString(", ", "[", "]")}")
            println("dependency parse: ${sentence.dependencyParse()}")
            println()
        }
    }

    @Test
    fun test_split_dependency_coref_en() {
        val props = Properties()
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,depparse,coref")
        props.setProperty("coref.algorithm", "neural")
        val pipeline = StanfordCoreNLP(props)
        val document = CoreDocument(en_text)
        pipeline.annotate(document)
        document.sentences().forEach { sentence ->
            println("sentence: ${sentence.text()}")
            println("tokens: ${sentence.tokens().map { "${it.originalText()} ${it.index()}"}.joinToString(", ", "[", "]")}")
            println("enitity mentions: ${sentence.entityMentions().map { "${it.text()} ${it.canonicalEntityMention().map { "${it.text()} - ${it.canonicalEntityMention()} ${it.entityType()} in \"${it.sentence()}\"" }}" }}")
            println()
        }
        document.corefChains().forEach { corefChain ->
            println("coref chains: $corefChain")
        }
    }

    @Test
    fun test_split_dependency_kbp_en() {
        val props = Properties()
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,depparse,kbp")
        props.setProperty("coref.algorithm", "neural")
        val pipeline = StanfordCoreNLP(props)
        val document = CoreDocument(en_text)
        pipeline.annotate(document)
        document.sentences().forEach { sentence ->
            println("sentence: ${sentence.text()}")
            println("tokens: ${sentence.tokens().map { "${it.originalText()} ${it.index()}"}.joinToString(", ", "[", "]")}")
            println("relations: ${sentence.relations()}")
            println()
        }
    }

    @Test
    fun test_split_dependency_quote_en() {
        val props = Properties()
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,depparse,coref,quote")
        props.setProperty("coref.algorithm", "neural")
        val pipeline = StanfordCoreNLP(props)
        val document = CoreDocument(en_text)
        pipeline.annotate(document)
        document.sentences().forEach { sentence ->
            println("sentence: ${sentence.text()}")
            println("tokens: ${sentence.tokens().map { "${it.originalText()} ${it.index()}"}.joinToString(", ", "[", "]")}")
        }
        document.quotes().forEach { quote ->
            println("quote by ${quote.speaker()} : ${quote.text()}")
        }
    }

    @Test
    fun test_split_dependency_sentiment_en() {
        val props = Properties()
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,depparse,sentiment")
        props.setProperty("coref.algorithm", "neural")
        val pipeline = StanfordCoreNLP(props)
        val document = CoreDocument(en_text)
        pipeline.annotate(document)
        document.sentences().forEach { sentence ->
            println("sentence: ${sentence.text()}")
            println("tokens: ${sentence.tokens().map { "${it.originalText()} ${it.index()}"}.joinToString(", ", "[", "]")}")
            println("sentiments: ${sentence.sentiment()}")
            println("sentimentsTree: ${sentence.sentimentTree()}")
            println()
        }
    }
}