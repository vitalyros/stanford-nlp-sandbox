package ru.vitalyros.sandbox.stanford.nlp

import edu.stanford.nlp.ling.CoreAnnotations
import edu.stanford.nlp.pipeline.CoreDocument
import edu.stanford.nlp.pipeline.StanfordCoreNLP
import org.junit.Test
import java.util.*


class TestRussian {

    val ru_text = "Джо Смит родился в Калифорнии. " +
            "Летом 2017-го он отправился в Париж во Францию. " +
            "Его рейс отправлялся в 3 часа дня 10-го июля 2017-го. " +
            "После того как Джо первый раз попробовал эскарго, он воскликнул \"Это было чудесно!\". " +
            "Он послал открытку своей сестре, Джейн Смит. " +
            "Услышав о поездке Джо, Джейн решила что однажды отправится во Францию."

    @Test
    fun test_split_ru() {
        val props = getProperties()
        props.setProperty("annotators", "tokenize,ssplit")
        val pipeline = StanfordCoreNLP(props)
        val document = CoreDocument(ru_text)
        pipeline.annotate(document)
        document.sentences().forEach { sentence ->
            println("sentence: ${sentence.text()}")
            println("tokens: ${sentence.tokens().map { "'${it.originalText()}'" }.joinToString(", ", "[", "]")}")
        }
    }

    @Test
    fun test_split_pos_ru() {
        val props = getProperties()
        props.setProperty("annotators", "tokenize,ssplit,pos")
        val pipeline = StanfordCoreNLP(props)
        val document = CoreDocument(ru_text)
        pipeline.annotate(document)
        document.sentences().forEach { sentence ->
            println("sentence: ${sentence.text()}")
            println("tokens: ${sentence.tokens().map { "'${it.originalText()}':${it.tag()}"}.joinToString(", ", "[", "]")}")
            println("pos tags: ${sentence.posTags()}")
            println()
        }
    }

    @Test
    fun test_split_lemma_en() {
        val props = getProperties()
        props.setProperty("annotators", "tokenize,ssplit,pos,custom.lemma")
        val pipeline = StanfordCoreNLP(props)
        val document = CoreDocument(ru_text)
        pipeline.annotate(document)
        document.sentences().forEach { sentence ->
            println("sentence: ${sentence.text()}")
            println("tokens: ${sentence.tokens().map { "'${it.originalText()}'/'${it.lemma()}'"}.joinToString(", ", "[", "]")}")
        }
    }

    @Test
    fun test_split_morpho_ru() {
        val props = getProperties()
        props.setProperty("annotators", "tokenize,ssplit,pos,custom.morpho")
        val pipeline = StanfordCoreNLP(props)
        val document = CoreDocument(ru_text)
        pipeline.annotate(document)
        document.sentences().forEach { sentence ->
            println("sentence: ${sentence.text()}")
            println("tokens: ${sentence.tokens().map {"${it.originalText()} ${it.get(CoreAnnotations.CoNLLUFeats::class.java)}} "}.joinToString(", ", "[", "]")}")
            println("core map: ${sentence.tokens()}")
        }
    }

    @Test
    fun test_split_depparse_ru() {
        val props = getProperties()
        props.setProperty("annotators", "tokenize,ssplit,pos,custom.lemma,custom.morpho,depparse")
        val pipeline = StanfordCoreNLP(props)
        val document = CoreDocument(ru_text)
        pipeline.annotate(document)
        document.sentences().forEach { sentence ->
            println("sentence: ${sentence.text()}")
            println("tokens: ${sentence.tokens().map { "${it.originalText()} ${it.index()} ${it.lemma()}"}.joinToString(", ", "[", "]")}")
            println("dependency parse: ${sentence.dependencyParse()}")
            println()
        }
    }


    fun getProperties() : Properties {
        val props = Properties()
        val stream = javaClass.classLoader.getResourceAsStream("StanfordCoreNLP-russian.properties")
        props.load(stream)
        stream.close()
        return props
    }
}