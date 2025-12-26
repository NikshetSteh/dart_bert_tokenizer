import 'dart:io';

import 'trie.dart';

class SpecialTokens {
  static const String pad = '[PAD]';
  static const String unk = '[UNK]';
  static const String cls = '[CLS]';
  static const String sep = '[SEP]';
  static const String mask = '[MASK]';
  static const List<String> defaults = [pad, unk, cls, sep, mask];
}

class Vocabulary {
  final Map<String, int> _tokenToId = {};
  final List<String> _idToToken = [];
  final Trie _trie = Trie();
  final Trie _subwordTrie = Trie();
  final String subwordPrefix;

  Vocabulary({this.subwordPrefix = '##'});

  int get size => _idToToken.length;

  Trie get trie => _trie;

  Trie get subwordTrie => _subwordTrie;

  int get unkTokenId => _tokenToId[SpecialTokens.unk] ?? 100;

  int get clsTokenId => _tokenToId[SpecialTokens.cls] ?? 101;

  int get sepTokenId => _tokenToId[SpecialTokens.sep] ?? 102;

  int get padTokenId => _tokenToId[SpecialTokens.pad] ?? 0;

  int get maskTokenId => _tokenToId[SpecialTokens.mask] ?? 103;

  static Future<Vocabulary> fromFile(
    String path, {
    String subwordPrefix = '##',
  }) async {
    final file = File(path);
    final lines = await file.readAsLines();
    return Vocabulary._fromLines(lines, subwordPrefix: subwordPrefix);
  }

  static Vocabulary fromFileSync(String path, {String subwordPrefix = '##'}) {
    final file = File(path);
    final lines = file.readAsLinesSync();
    return Vocabulary._fromLines(lines, subwordPrefix: subwordPrefix);
  }

  static Vocabulary fromString(String content, {String subwordPrefix = '##'}) {
    final lines = content.split('\n');
    return Vocabulary._fromLines(lines, subwordPrefix: subwordPrefix);
  }

  static Vocabulary fromTokens(
    List<String> tokens, {
    String subwordPrefix = '##',
  }) {
    return Vocabulary._fromLines(tokens, subwordPrefix: subwordPrefix);
  }

  static Vocabulary _fromLines(
    List<String> lines, {
    String subwordPrefix = '##',
  }) {
    final vocab = Vocabulary(subwordPrefix: subwordPrefix);

    for (var i = 0; i < lines.length; i++) {
      final token = lines[i].trim();
      if (token.isEmpty) continue;

      vocab._addToken(token, i);
    }

    return vocab;
  }

  void _addToken(String token, int id) {
    _tokenToId[token] = id;

    while (_idToToken.length <= id) {
      _idToToken.add('');
    }
    _idToToken[id] = token;

    if (token.startsWith(subwordPrefix)) {
      final subword = token.substring(subwordPrefix.length);
      _subwordTrie.insert(subword, id);
    } else if (!token.startsWith('[') || !token.endsWith(']')) {
      _trie.insert(token, id);
    }
  }

  int tokenToId(String token) {
    return _tokenToId[token] ?? unkTokenId;
  }

  String idToToken(int id) {
    if (id < 0 || id >= _idToToken.length) {
      return SpecialTokens.unk;
    }
    final token = _idToToken[id];
    return token.isEmpty ? SpecialTokens.unk : token;
  }

  bool contains(String token) => _tokenToId.containsKey(token);

  bool isSpecialToken(String token) {
    return token.startsWith('[') && token.endsWith(']');
  }

  Map<String, int> get vocabularyMap => Map.unmodifiable(_tokenToId);

  List<String> get tokens => List.unmodifiable(_idToToken);

  TrieMatch? findLongestMatch(
    String text, {
    int startIndex = 0,
    bool isSubword = false,
  }) {
    final targetTrie = isSubword ? _subwordTrie : _trie;
    return targetTrie.findLongestPrefix(text, startIndex);
  }

  @override
  String toString() => 'Vocabulary(size: $size)';
}
