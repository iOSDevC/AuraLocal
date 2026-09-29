import XCTest
@testable import AuraCore

final class NLToolsTests: XCTestCase {

    private static let spanishSentence = "El perro corre por el parque mientras los niños juegan con la pelota."
    private static let englishSentence = "The dog runs through the park while the children play with the ball."

    // MARK: - Metadata

    func testLanguageGroupIdsAndCategory() {
        let tools: [any SystemTool] = [NLLanguageIdentificationTool(), NLEntityRecognitionTool(), NLSentimentTool()]
        XCTAssertEqual(tools.map(\.id), ["system.nl.language", "system.nl.entities", "system.nl.sentiment"])
        XCTAssertTrue(tools.allSatisfy { $0.category == .language })
        XCTAssertTrue(tools.allSatisfy { !$0.summary.isEmpty && !$0.displayName.isEmpty })
    }

    func testLanguageGroupToolsAreAvailableOnThisHost() async {
        let language = await NLLanguageIdentificationTool().availability()
        let entities = await NLEntityRecognitionTool().availability()
        let sentiment = await NLSentimentTool().availability()
        XCTAssertEqual(language, .available)
        XCTAssertEqual(entities, .available)
        XCTAssertEqual(sentiment, .available)
    }

    // MARK: - Language identification

    func testIdentifiesSpanish() {
        let result = NLLanguageIdentificationTool().identify(Self.spanishSentence)
        XCTAssertEqual(result.dominantLanguage, "es")
        XCTAssertEqual(result.hypotheses.first?.language, "es")
    }

    func testIdentifiesEnglish() {
        let result = NLLanguageIdentificationTool().identify(Self.englishSentence)
        XCTAssertEqual(result.dominantLanguage, "en")
        XCTAssertEqual(result.hypotheses.first?.language, "en")
    }

    func testHypothesesAreBoundedProbabilitiesSortedDescending() {
        let tool = NLLanguageIdentificationTool()
        for text in [Self.spanishSentence, Self.englishSentence, "Hola", "ok"] {
            let hypotheses = tool.identify(text, maxHypotheses: 5).hypotheses
            XCTAssertFalse(hypotheses.isEmpty, text)
            XCTAssertLessThanOrEqual(hypotheses.count, 5, text)
            XCTAssertTrue(hypotheses.allSatisfy { (0...1).contains($0.probability) }, text)
            XCTAssertLessThanOrEqual(hypotheses.map(\.probability).reduce(0, +), 1 + 1e-6, text)
            XCTAssertEqual(hypotheses.map(\.probability), hypotheses.map(\.probability).sorted(by: >), text)
        }
    }

    func testConstraintsRestrictTheAnswer() {
        let allowed: Set<String> = ["en", "fr"]
        let result = NLLanguageIdentificationTool().identify(Self.spanishSentence, constraints: Array(allowed))
        XCTAssertTrue(allowed.contains(result.dominantLanguage ?? ""), "got \(String(describing: result.dominantLanguage))")
        XCTAssertFalse(result.hypotheses.isEmpty)
        XCTAssertTrue(result.hypotheses.allSatisfy { allowed.contains($0.language) }, "\(result.hypotheses)")
    }

    func testSingleLanguageConstraintForcesThatLanguage() {
        let result = NLLanguageIdentificationTool().identify(Self.englishSentence, constraints: ["es"])
        XCTAssertEqual(result.dominantLanguage, "es")
    }

    func testEmptyOrDigitsOnlyTextIsUndetermined() {
        let tool = NLLanguageIdentificationTool()
        for text in ["", "   ", "12345 !!!"] {
            let result = tool.identify(text)
            XCTAssertNil(result.dominantLanguage, text)
            XCTAssertTrue(result.hypotheses.isEmpty, text)
        }
    }

    func testZeroMaxHypothesesReturnsNoneButKeepsDominant() {
        let result = NLLanguageIdentificationTool().identify(Self.englishSentence, maxHypotheses: 0)
        XCTAssertEqual(result.dominantLanguage, "en")
        XCTAssertTrue(result.hypotheses.isEmpty)
    }

    // MARK: - Named entities

    func testEntitiesInEnglish() throws {
        let tool = NLEntityRecognitionTool()
        try XCTSkipUnless(tool.supports(language: "en"), "No English name model on this host.")
        let text = "Tim Cook announced in Cupertino that Apple will open a new office in London."
        let found = tool.entities(in: text)
        XCTAssertEqual(found.filter { $0.kind == .person }.map(\.text), ["Tim Cook"])
        XCTAssertEqual(found.filter { $0.kind == .place }.map(\.text), ["Cupertino", "London"])
        XCTAssertEqual(found.filter { $0.kind == .organization }.map(\.text), ["Apple"])
        for entity in found {
            XCTAssertEqual((text as NSString).substring(with: entity.range), entity.text)
        }
    }

    func testEntitiesInSpanish() throws {
        let tool = NLEntityRecognitionTool()
        try XCTSkipUnless(tool.supports(language: "es"), "No Spanish name model on this host.")
        let text = "Pedro Sánchez se reunió en Madrid con representantes de Telefónica y del Banco Santander."
        for hint in [nil, "es"] {
            let found = tool.entities(in: text, language: hint)
            XCTAssertEqual(found.filter { $0.kind == .person }.map(\.text), ["Pedro Sánchez"])
            XCTAssertEqual(found.filter { $0.kind == .place }.map(\.text), ["Madrid"])
            XCTAssertEqual(found.filter { $0.kind == .organization }.map(\.text), ["Telefónica", "Banco Santander"])
        }
    }

    func testSpanishMultiWordNamesAndPlaces() throws {
        let tool = NLEntityRecognitionTool()
        try XCTSkipUnless(tool.supports(language: "es"), "No Spanish name model on this host.")
        let text = "Gabriel García Márquez nació en Aracataca, Colombia, y trabajó para el periódico El Espectador."
        let found = tool.entities(in: text, language: "es")
        XCTAssertEqual(found.filter { $0.kind == .person }.map(\.text), ["Gabriel García Márquez"])
        XCTAssertEqual(found.filter { $0.kind == .place }.map(\.text), ["Aracataca", "Colombia"])
        XCTAssertTrue(found.contains { $0.kind == .organization && $0.text.hasSuffix("El Espectador") })
    }

    func testEntityRangesAreUTF16Offsets() throws {
        let tool = NLEntityRecognitionTool()
        try XCTSkipUnless(tool.supports(language: "es"), "No Spanish name model on this host.")
        let text = "🙂 José Martí vivió en Nueva York."
        let found = tool.entities(in: text, language: "es")
        let person = try XCTUnwrap(found.first { $0.kind == .person })
        XCTAssertEqual(person.text, "José Martí")
        XCTAssertEqual(person.range, NSRange(location: 3, length: 10))
        XCTAssertEqual(found.first { $0.kind == .place }?.text, "Nueva York")
    }

    func testTextWithoutNamesHasNoEntities() {
        XCTAssertTrue(NLEntityRecognitionTool().entities(in: "the quick brown fox jumps over the lazy dog").isEmpty)
        XCTAssertTrue(NLEntityRecognitionTool().entities(in: "").isEmpty)
    }

    // MARK: - Sentiment

    func testSentimentSupportsEnglishAndSpanishButNotJapanese() {
        let tool = NLSentimentTool()
        XCTAssertTrue(tool.supports(language: "en"))
        XCTAssertTrue(tool.supports(language: "es"))
        XCTAssertFalse(tool.supports(language: "ja"))
    }

    func testSentimentSignForClearEnglish() throws {
        let tool = NLSentimentTool()
        let positive = try XCTUnwrap(tool.score("I love this product, it is absolutely wonderful and works perfectly."))
        let negative = try XCTUnwrap(tool.score("This is the worst experience ever. I hate it, it is terrible and broken."))
        XCTAssertGreaterThan(positive, 0.5)
        XCTAssertLessThan(negative, -0.5)
        XCTAssertTrue((-1...1).contains(positive) && (-1...1).contains(negative))
    }

    func testSentimentSignForClearSpanish() throws {
        let tool = NLSentimentTool()
        for hint in [nil, "es"] {
            let positive = try XCTUnwrap(tool.score(
                "Me encanta este producto, es absolutamente maravilloso y funciona perfectamente.", language: hint))
            let negative = try XCTUnwrap(tool.score(
                "Esta es la peor experiencia de mi vida. Lo odio, es horrible y está roto.", language: hint))
            XCTAssertGreaterThan(positive, 0.5)
            XCTAssertLessThan(negative, -0.5)
        }
    }

    func testSentimentUnsupportedLanguageIsNil() {
        let tool = NLSentimentTool()
        XCTAssertNil(tool.score("この製品が大好きです。本当に素晴らしいです。"))
        XCTAssertNil(tool.score("I love this product, it is wonderful.", language: "ja"))
    }

    func testSentimentBlankTextIsNil() {
        XCTAssertNil(NLSentimentTool().score(""))
        XCTAssertNil(NLSentimentTool().score("   \n\n  "))
        XCTAssertNil(NLSentimentTool().score("   ", language: "en"))
    }

    func testSentimentIgnoresBlankParagraphs() {
        let tool = NLSentimentTool()
        let single = "I love this product, it is wonderful."
        XCTAssertEqual(tool.score(single + "\n\n\n\n"), tool.score(single))
    }
}
