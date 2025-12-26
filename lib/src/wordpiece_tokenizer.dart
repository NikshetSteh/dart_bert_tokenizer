import 'dart:async';
import 'dart:isolate';

import 'encoding.dart';
import 'pre_tokenizer.dart';
import 'vocabulary.dart';

const _kMinBatchSizeForParallel = 8;

enum PaddingDirection { right, left }

class PaddingConfig {
  final PaddingDirection direction;
  final int? length;
  final int? padToMultipleOf;

  const PaddingConfig({
    this.direction = PaddingDirection.right,
    this.length,
    this.padToMultipleOf,
  });
}

enum TruncationDirection { right, left }

class TruncationConfig {
  final int maxLength;
  final TruncationDirection direction;
  final TruncationStrategy strategy;

  const TruncationConfig({
    required this.maxLength,
    this.direction = TruncationDirection.right,
    this.strategy = TruncationStrategy.longestFirst,
  });
}

class WordPieceConfig {
  final bool lowercase;
  final bool stripAccents;
  final bool handleChineseChars;
  final String subwordPrefix;
  final int maxWordLength;
  final bool addClsToken;
  final bool addSepToken;

  const WordPieceConfig({
    this.lowercase = true,
    this.stripAccents = true,
    this.handleChineseChars = true,
    this.subwordPrefix = '##',
    this.maxWordLength = 200,
    this.addClsToken = true,
    this.addSepToken = true,
  });
}

class WordPieceTokenizer {
  final Vocabulary vocab;
  final WordPieceConfig config;
  late final BertPreTokenizer _preTokenizer;
  PaddingConfig? _paddingConfig;
  TruncationConfig? _truncationConfig;

  WordPieceTokenizer({
    required this.vocab,
    this.config = const WordPieceConfig(),
  }) {
    _preTokenizer = BertPreTokenizer(
      lowercase: config.lowercase,
      stripAccents: config.stripAccents,
      handleChineseChars: config.handleChineseChars,
    );
  }

  PaddingConfig? get padding => _paddingConfig;

  TruncationConfig? get truncation => _truncationConfig;

  WordPieceTokenizer enablePadding({
    PaddingDirection direction = PaddingDirection.right,
    int? length,
    int? padToMultipleOf,
  }) {
    _paddingConfig = PaddingConfig(
      direction: direction,
      length: length,
      padToMultipleOf: padToMultipleOf,
    );
    return this;
  }

  WordPieceTokenizer noPadding() {
    _paddingConfig = null;
    return this;
  }

  WordPieceTokenizer enableTruncation({
    required int maxLength,
    TruncationDirection direction = TruncationDirection.right,
    TruncationStrategy strategy = TruncationStrategy.longestFirst,
  }) {
    _truncationConfig = TruncationConfig(
      maxLength: maxLength,
      direction: direction,
      strategy: strategy,
    );
    return this;
  }

  WordPieceTokenizer noTruncation() {
    _truncationConfig = null;
    return this;
  }

  Encoding _applyPostProcessing(Encoding encoding) {
    var result = encoding;

    if (_truncationConfig != null) {
      result = result.withTruncation(
        maxLength: _truncationConfig!.maxLength,
        truncateFromEnd:
            _truncationConfig!.direction == TruncationDirection.right,
      );
    }

    if (_paddingConfig != null) {
      final padOnRight = _paddingConfig!.direction == PaddingDirection.right;

      if (_paddingConfig!.length != null) {
        result = result.withPadding(
          targetLength: _paddingConfig!.length!,
          padTokenId: vocab.padTokenId,
          padOnRight: padOnRight,
        );
      }

      if (_paddingConfig!.padToMultipleOf != null) {
        result = result.withPaddingToMultipleOf(
          multiple: _paddingConfig!.padToMultipleOf!,
          padTokenId: vocab.padTokenId,
          padOnRight: padOnRight,
        );
      }
    }

    return result;
  }

  List<Encoding> _applyBatchPostProcessing(List<Encoding> encodings) {
    if (encodings.isEmpty) return encodings;

    var results = encodings;
    if (_truncationConfig != null) {
      results = results
          .map(
            (e) => e.withTruncation(
              maxLength: _truncationConfig!.maxLength,
              truncateFromEnd:
                  _truncationConfig!.direction == TruncationDirection.right,
            ),
          )
          .toList();
    }

    if (_paddingConfig != null) {
      final padOnRight = _paddingConfig!.direction == PaddingDirection.right;

      int targetLength;
      if (_paddingConfig!.length != null) {
        targetLength = _paddingConfig!.length!;
      } else {
        targetLength = results
            .map((e) => e.length)
            .reduce((a, b) => a > b ? a : b);
      }

      if (_paddingConfig!.padToMultipleOf != null) {
        final multiple = _paddingConfig!.padToMultipleOf!;
        final remainder = targetLength % multiple;
        if (remainder != 0) {
          targetLength += multiple - remainder;
        }
      }

      results = results
          .map(
            (e) => e.withPadding(
              targetLength: targetLength,
              padTokenId: vocab.padTokenId,
              padOnRight: padOnRight,
            ),
          )
          .toList();
    }

    return results;
  }

  static Future<WordPieceTokenizer> fromVocabFile(
    String path, {
    WordPieceConfig config = const WordPieceConfig(),
  }) async {
    final vocab = await Vocabulary.fromFile(
      path,
      subwordPrefix: config.subwordPrefix,
    );
    return WordPieceTokenizer(vocab: vocab, config: config);
  }

  static WordPieceTokenizer fromVocabFileSync(
    String path, {
    WordPieceConfig config = const WordPieceConfig(),
  }) {
    final vocab = Vocabulary.fromFileSync(
      path,
      subwordPrefix: config.subwordPrefix,
    );
    return WordPieceTokenizer(vocab: vocab, config: config);
  }

  int numSpecialTokensToAdd({bool isPair = false}) {
    var count = 0;
    if (config.addClsToken) count++;
    if (config.addSepToken) count++;
    if (isPair && config.addSepToken) count++;
    return count;
  }

  Encoding encode(String text, {bool? addSpecialTokens}) {
    final shouldAddCls = addSpecialTokens ?? config.addClsToken;
    final shouldAddSep = addSpecialTokens ?? config.addSepToken;

    final builder = EncodingBuilder();

    if (shouldAddCls) {
      builder.addSpecialToken(
        token: SpecialTokens.cls,
        id: vocab.clsTokenId,
        typeId: 0,
      );
    }

    final preTokens = _preTokenizer.preTokenize(text);

    for (var wordIdx = 0; wordIdx < preTokens.length; wordIdx++) {
      final preToken = preTokens[wordIdx];
      final wordTokens = _tokenizeWord(preToken.text);

      for (final tokenInfo in wordTokens) {
        builder.addToken(
          token: tokenInfo.token,
          id: tokenInfo.id,
          typeId: 0,
          offset: (
            preToken.start + tokenInfo.startOffset,
            preToken.start + tokenInfo.endOffset,
          ),
          wordId: wordIdx,
        );
      }
    }

    if (shouldAddSep) {
      builder.addSpecialToken(
        token: SpecialTokens.sep,
        id: vocab.sepTokenId,
        typeId: 0,
      );
    }

    return _applyPostProcessing(builder.build());
  }

  Encoding encodePair(
    String textA,
    String textB, {
    bool? addSpecialTokens,
    int? maxLength,
    TruncationStrategy truncationStrategy = TruncationStrategy.longestFirst,
  }) {
    final shouldAddCls = addSpecialTokens ?? config.addClsToken;
    final shouldAddSep = addSpecialTokens ?? config.addSepToken;

    final encodingA = encode(textA, addSpecialTokens: false);
    final encodingB = encode(textB, addSpecialTokens: false);

    final effectiveMaxLength = maxLength ?? _truncationConfig?.maxLength;
    final effectiveStrategy = _truncationConfig?.strategy ?? truncationStrategy;

    final (truncatedA, truncatedB) = effectiveMaxLength != null
        ? Encoding.truncatePair(
            encodingA: encodingA,
            encodingB: encodingB,
            maxLength: effectiveMaxLength,
            strategy: effectiveStrategy,
            numSpecialTokens: numSpecialTokensToAdd(isPair: true),
          )
        : (encodingA, encodingB);

    final builder = EncodingBuilder();

    if (shouldAddCls) {
      builder.addSpecialToken(
        token: SpecialTokens.cls,
        id: vocab.clsTokenId,
        typeId: 0,
      );
    }

    for (var i = 0; i < truncatedA.length; i++) {
      builder.addToken(
        token: truncatedA.tokens[i],
        id: truncatedA.ids[i],
        typeId: 0,
        offset: truncatedA.offsets[i],
        wordId: truncatedA.wordIds[i],
      );
    }

    if (shouldAddSep) {
      builder.addSpecialToken(
        token: SpecialTokens.sep,
        id: vocab.sepTokenId,
        typeId: 0,
      );
    }

    final wordIdOffset = truncatedA.wordIds.where((id) => id != null).length;
    for (var i = 0; i < truncatedB.length; i++) {
      final originalWordId = truncatedB.wordIds[i];
      builder.addToken(
        token: truncatedB.tokens[i],
        id: truncatedB.ids[i],
        typeId: 1,
        offset: truncatedB.offsets[i],
        wordId: originalWordId != null ? wordIdOffset + originalWordId : null,
      );
    }

    if (shouldAddSep) {
      builder.addSpecialToken(
        token: SpecialTokens.sep,
        id: vocab.sepTokenId,
        typeId: 1,
      );
    }

    var result = builder.build();
    if (_paddingConfig != null) {
      final padOnRight = _paddingConfig!.direction == PaddingDirection.right;

      if (_paddingConfig!.length != null) {
        result = result.withPadding(
          targetLength: _paddingConfig!.length!,
          padTokenId: vocab.padTokenId,
          padOnRight: padOnRight,
        );
      }

      if (_paddingConfig!.padToMultipleOf != null) {
        result = result.withPaddingToMultipleOf(
          multiple: _paddingConfig!.padToMultipleOf!,
          padTokenId: vocab.padTokenId,
          padOnRight: padOnRight,
        );
      }
    }

    return result;
  }

  List<Encoding> encodeBatch(List<String> texts, {bool? addSpecialTokens}) {
    final savedPadding = _paddingConfig;
    final savedTruncation = _truncationConfig;
    _paddingConfig = null;
    _truncationConfig = null;

    final encodings = texts
        .map((text) => encode(text, addSpecialTokens: addSpecialTokens))
        .toList();

    _paddingConfig = savedPadding;
    _truncationConfig = savedTruncation;

    return _applyBatchPostProcessing(encodings);
  }

  Future<List<Encoding>> encodeBatchParallel(
    List<String> texts, {
    bool? addSpecialTokens,
    int? numWorkers,
  }) async {
    if (texts.length < _kMinBatchSizeForParallel) {
      return encodeBatch(texts, addSpecialTokens: addSpecialTokens);
    }

    final workerCount = numWorkers ?? _getOptimalWorkerCount(texts.length);
    final chunkSize = (texts.length / workerCount).ceil();

    final vocabTokens = vocab.tokens;
    final futures = <Future<List<_EncodingData>>>[];

    for (var i = 0; i < workerCount; i++) {
      final start = i * chunkSize;
      if (start >= texts.length) break;

      final end = (start + chunkSize).clamp(0, texts.length);
      final chunk = texts.sublist(start, end);

      futures.add(
        Isolate.run(
          () => _encodeChunkInIsolate(
            chunk,
            vocabTokens,
            config,
            addSpecialTokens,
          ),
        ),
      );
    }

    final results = await Future.wait(futures);

    final encodings = <Encoding>[];
    for (final chunkResults in results) {
      for (final data in chunkResults) {
        encodings.add(data.toEncoding());
      }
    }

    return _applyBatchPostProcessing(encodings);
  }

  Future<List<Encoding>> encodePairBatchParallel(
    List<(String, String)> pairs, {
    bool? addSpecialTokens,
    int? maxLength,
    TruncationStrategy truncationStrategy = TruncationStrategy.longestFirst,
    int? numWorkers,
  }) async {
    if (pairs.length < _kMinBatchSizeForParallel) {
      return encodePairBatch(
        pairs,
        addSpecialTokens: addSpecialTokens,
        maxLength: maxLength,
        truncationStrategy: truncationStrategy,
      );
    }

    final workerCount = numWorkers ?? _getOptimalWorkerCount(pairs.length);
    final chunkSize = (pairs.length / workerCount).ceil();

    final vocabTokens = vocab.tokens;
    final effectiveMaxLength = maxLength ?? _truncationConfig?.maxLength;
    final futures = <Future<List<_EncodingData>>>[];

    for (var i = 0; i < workerCount; i++) {
      final start = i * chunkSize;
      if (start >= pairs.length) break;

      final end = (start + chunkSize).clamp(0, pairs.length);
      final chunk = pairs.sublist(start, end);
      final chunkData = chunk.map((p) => [p.$1, p.$2]).toList();
      futures.add(
        Isolate.run(
          () => _encodePairChunkInIsolate(
            chunkData,
            vocabTokens,
            config,
            addSpecialTokens,
            effectiveMaxLength,
            truncationStrategy,
          ),
        ),
      );
    }

    final results = await Future.wait(futures);
    final encodings = <Encoding>[];
    for (final chunkResults in results) {
      for (final data in chunkResults) {
        encodings.add(data.toEncoding());
      }
    }

    if (_paddingConfig != null) {
      final padOnRight = _paddingConfig!.direction == PaddingDirection.right;

      int targetLength;
      if (_paddingConfig!.length != null) {
        targetLength = _paddingConfig!.length!;
      } else {
        targetLength = encodings
            .map((e) => e.length)
            .reduce((a, b) => a > b ? a : b);
      }

      if (_paddingConfig!.padToMultipleOf != null) {
        final multiple = _paddingConfig!.padToMultipleOf!;
        final remainder = targetLength % multiple;
        if (remainder != 0) {
          targetLength += multiple - remainder;
        }
      }

      return encodings
          .map(
            (e) => e.withPadding(
              targetLength: targetLength,
              padTokenId: vocab.padTokenId,
              padOnRight: padOnRight,
            ),
          )
          .toList();
    }

    return encodings;
  }

  int _getOptimalWorkerCount(int batchSize) {
    const maxWorkers = 4;
    const minItemsPerWorker = 4;

    final workersByItems = (batchSize / minItemsPerWorker).floor();
    return workersByItems.clamp(1, maxWorkers);
  }

  List<Encoding> encodePairBatch(
    List<(String, String)> pairs, {
    bool? addSpecialTokens,
    int? maxLength,
    TruncationStrategy truncationStrategy = TruncationStrategy.longestFirst,
  }) {
    final savedPadding = _paddingConfig;
    _paddingConfig = null;

    final encodings = pairs
        .map(
          (pair) => encodePair(
            pair.$1,
            pair.$2,
            addSpecialTokens: addSpecialTokens,
            maxLength: maxLength,
            truncationStrategy: truncationStrategy,
          ),
        )
        .toList();

    _paddingConfig = savedPadding;

    if (_paddingConfig != null) {
      final padOnRight = _paddingConfig!.direction == PaddingDirection.right;

      int targetLength;
      if (_paddingConfig!.length != null) {
        targetLength = _paddingConfig!.length!;
      } else {
        targetLength = encodings
            .map((e) => e.length)
            .reduce((a, b) => a > b ? a : b);
      }

      if (_paddingConfig!.padToMultipleOf != null) {
        final multiple = _paddingConfig!.padToMultipleOf!;
        final remainder = targetLength % multiple;
        if (remainder != 0) {
          targetLength += multiple - remainder;
        }
      }

      return encodings
          .map(
            (e) => e.withPadding(
              targetLength: targetLength,
              padTokenId: vocab.padTokenId,
              padOnRight: padOnRight,
            ),
          )
          .toList();
    }

    return encodings;
  }

  List<_TokenInfo> _tokenizeWord(String word) {
    if (word.isEmpty) {
      return [];
    }

    if (word.length > config.maxWordLength) {
      return [
        _TokenInfo(
          token: SpecialTokens.unk,
          id: vocab.unkTokenId,
          startOffset: 0,
          endOffset: word.length,
        ),
      ];
    }

    final tokens = <_TokenInfo>[];
    var start = 0;
    var isFirstSubword = true;

    while (start < word.length) {
      final match = _findLongestMatchAt(
        word,
        start,
        isSubword: !isFirstSubword,
      );

      if (match == null) {
        return [
          _TokenInfo(
            token: SpecialTokens.unk,
            id: vocab.unkTokenId,
            startOffset: 0,
            endOffset: word.length,
          ),
        ];
      }

      final matchedText = word.substring(start, match.endIndex);
      final tokenStr = isFirstSubword
          ? matchedText
          : '${config.subwordPrefix}$matchedText';

      tokens.add(
        _TokenInfo(
          token: tokenStr,
          id: match.tokenId,
          startOffset: start,
          endOffset: match.endIndex,
        ),
      );

      start = match.endIndex;
      isFirstSubword = false;
    }

    return tokens;
  }

  _TrieMatchResult? _findLongestMatchAt(
    String word,
    int startIndex, {
    required bool isSubword,
  }) {
    final trie = isSubword ? vocab.subwordTrie : vocab.trie;
    final match = trie.findLongestPrefix(word, startIndex);
    if (match != null) {
      return _TrieMatchResult(endIndex: match.end, tokenId: match.tokenId);
    }

    if (startIndex < word.length) {
      final charCode = word.codeUnitAt(startIndex);
      final firstChar = String.fromCharCode(charCode);
      final tokenToCheck = isSubword
          ? '${config.subwordPrefix}$firstChar'
          : firstChar;

      if (vocab.contains(tokenToCheck)) {
        return _TrieMatchResult(
          endIndex: startIndex + 1,
          tokenId: vocab.tokenToId(tokenToCheck),
        );
      }
    }

    return null;
  }

  String decode(List<int> ids, {bool skipSpecialTokens = true}) {
    final buffer = StringBuffer();
    var isFirst = true;

    for (final id in ids) {
      final token = vocab.idToToken(id);

      if (skipSpecialTokens && vocab.isSpecialToken(token)) {
        continue;
      }

      if (token.startsWith(config.subwordPrefix)) {
        buffer.write(token.substring(config.subwordPrefix.length));
      } else {
        if (!isFirst) {
          buffer.write(' ');
        }
        buffer.write(token);
        isFirst = false;
      }
    }

    return buffer.toString();
  }

  List<String> decodeBatch(
    List<List<int>> idsBatch, {
    bool skipSpecialTokens = true,
  }) {
    return idsBatch
        .map((ids) => decode(ids, skipSpecialTokens: skipSpecialTokens))
        .toList();
  }

  List<int> convertTokensToIds(List<String> tokens) {
    return tokens.map(vocab.tokenToId).toList();
  }

  List<String> convertIdsToTokens(List<int> ids) {
    return ids.map(vocab.idToToken).toList();
  }
}

class _TokenInfo {
  final String token;
  final int id;
  final int startOffset;
  final int endOffset;

  const _TokenInfo({
    required this.token,
    required this.id,
    required this.startOffset,
    required this.endOffset,
  });
}

class _TrieMatchResult {
  final int endIndex;
  final int tokenId;

  const _TrieMatchResult({required this.endIndex, required this.tokenId});
}

class _EncodingData {
  final List<String> tokens;
  final List<int> ids;
  final List<int> typeIds;
  final List<int> attentionMask;
  final List<int> specialTokensMask;
  final List<List<int>> offsets;
  final List<int?> wordIds;
  final List<int?> sequenceIds;

  const _EncodingData({
    required this.tokens,
    required this.ids,
    required this.typeIds,
    required this.attentionMask,
    required this.specialTokensMask,
    required this.offsets,
    required this.wordIds,
    required this.sequenceIds,
  });

  factory _EncodingData.fromEncoding(Encoding encoding) {
    return _EncodingData(
      tokens: encoding.tokens.toList(),
      ids: encoding.ids.toList(),
      typeIds: encoding.typeIds.toList(),
      attentionMask: encoding.attentionMask.toList(),
      specialTokensMask: encoding.specialTokensMask.toList(),
      offsets: encoding.offsets.map((o) => [o.$1, o.$2]).toList(),
      wordIds: encoding.wordIds.toList(),
      sequenceIds: encoding.sequenceIds.toList(),
    );
  }

  Encoding toEncoding() {
    return Encoding(
      tokens: tokens,
      ids: ids,
      typeIds: typeIds,
      attentionMask: attentionMask,
      specialTokensMask: specialTokensMask,
      offsets: offsets.map((o) => (o[0], o[1])).toList(),
      wordIds: wordIds,
      sequenceIds: sequenceIds,
    );
  }
}

List<_EncodingData> _encodeChunkInIsolate(
  List<String> texts,
  List<String> vocabTokens,
  WordPieceConfig config,
  bool? addSpecialTokens,
) {
  final vocab = Vocabulary.fromTokens(
    vocabTokens,
    subwordPrefix: config.subwordPrefix,
  );
  final tokenizer = WordPieceTokenizer(vocab: vocab, config: config);
  final results = <_EncodingData>[];
  for (final text in texts) {
    final encoding = tokenizer.encode(text, addSpecialTokens: addSpecialTokens);
    results.add(_EncodingData.fromEncoding(encoding));
  }

  return results;
}

List<_EncodingData> _encodePairChunkInIsolate(
  List<List<String>> pairs,
  List<String> vocabTokens,
  WordPieceConfig config,
  bool? addSpecialTokens,
  int? maxLength,
  TruncationStrategy truncationStrategy,
) {
  final vocab = Vocabulary.fromTokens(
    vocabTokens,
    subwordPrefix: config.subwordPrefix,
  );
  final tokenizer = WordPieceTokenizer(vocab: vocab, config: config);
  final results = <_EncodingData>[];
  for (final pair in pairs) {
    final encoding = tokenizer.encodePair(
      pair[0],
      pair[1],
      addSpecialTokens: addSpecialTokens,
      maxLength: maxLength,
      truncationStrategy: truncationStrategy,
    );
    results.add(_EncodingData.fromEncoding(encoding));
  }

  return results;
}
