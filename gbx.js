/*
 * Copyright 2017 Sam Thorogood. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

/**
 * @fileoverview Polyfill for TextEncoder and TextDecoder.
 *
 * You probably want `text.min.js`, and not this file directly.
 */
var TextDecoder = Object;
 (function(scope) {
    
    // fail early
    if (TextDecoder) {
      return false;
    }
    
    /**
     * @constructor
     * @param {string=} utfLabel
     * @param {{fatal: boolean}=} options
     */
    function FastTextDecoder(utfLabel, options) { // (utfLabel='utf-8', options={fatal: false}) {
      if (validUtfLabels.indexOf(utfLabel.toLowerCase()) === -1) {
        // throw new RangeError(
        //   `Failed to construct 'TextDecoder': The encoding label provided ('${utfLabel}') is invalid.`);
      }
      if (options.fatal) {
        // throw new Error(`Failed to construct 'TextDecoder': the 'fatal' option is unsupported.`);
      }
    }
    
    Object.defineProperty(FastTextDecoder.prototype, 'encoding', {value: 'utf-8'});
    
    Object.defineProperty(FastTextDecoder.prototype, 'fatal', {value: false});
    
    Object.defineProperty(FastTextDecoder.prototype, 'ignoreBOM', {value: false});
    
    /**
     * @param {!Uint8Array} bytes
     * @return {string}
     */
    function decodeBuffer(bytes) {
      return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('utf-8');
    }
    
    /**
     * @param {!Uint8Array} bytes
     * @return {string}
     */
    function decodeSyncXHR(bytes) {
      let u;
    
      // This hack will fail in non-Edgium Edge because sync XHRs are disabled (and
      // possibly in other places), so ensure there's a fallback call.
      try {
        const b = new Blob([bytes], {type: 'text/plain;charset=UTF-8'});
        u = URL.createObjectURL(b);
    
        const x = new XMLHttpRequest();
        x.open('GET', u, false);
        x.send();
        return x.responseText;
      } catch (e) {
        return decodeFallback(bytes);
      } finally {
        if (u) {
          URL.revokeObjectURL(u);
        }
      }
    }
    
    /**
     * @param {!Uint8Array} bytes
     * @return {string}
     */
    function decodeFallback(bytes) {
      let inputIndex = 0;
    
      // Create a working buffer for UTF-16 code points, but don't generate one
      // which is too large for small input sizes. UTF-8 to UCS-16 conversion is
      // going to be at most 1:1, if all code points are ASCII. The other extreme
      // is 4-byte UTF-8, which results in two UCS-16 points, but this is still 50%
      // fewer entries in the output.
      const pendingSize = Math.min(256 * 256, bytes.length + 1);
      const pending = new Uint16Array(pendingSize);
      const chunks = [];
      let pendingIndex = 0;
    
      for (;;) {
        const more = inputIndex < bytes.length;
    
        // If there's no more data or there'd be no room for two UTF-16 values,
        // create a chunk. This isn't done at the end by simply slicing the data
        // into equal sized chunks as we might hit a surrogate pair.
        if (!more || (pendingIndex >= pendingSize - 1)) {
          // nb. .apply and friends are *really slow*. Low-hanging fruit is to
          // expand this to literally pass pending[0], pending[1], ... etc, but
          // the output code expands pretty fast in this case.
          chunks.push(String.fromCharCode.apply(null, pending.subarray(0, pendingIndex)));
    
          if (!more) {
            return chunks.join('');
          }
    
          // Move the buffer forward and create another chunk.
          bytes = bytes.subarray(inputIndex);
          inputIndex = 0;
          pendingIndex = 0;
        }
    
        // The native TextDecoder will generate "REPLACEMENT CHARACTER" where the
        // input data is invalid. Here, we blindly parse the data even if it's
        // wrong: e.g., if a 3-byte sequence doesn't have two valid continuations.
    
        const byte1 = bytes[inputIndex++];
        if ((byte1 & 0x80) === 0) {  // 1-byte or null
          pending[pendingIndex++] = byte1;
        } else if ((byte1 & 0xe0) === 0xc0) {  // 2-byte
          const byte2 = bytes[inputIndex++] & 0x3f;
          pending[pendingIndex++] = ((byte1 & 0x1f) << 6) | byte2;
        } else if ((byte1 & 0xf0) === 0xe0) {  // 3-byte
          const byte2 = bytes[inputIndex++] & 0x3f;
          const byte3 = bytes[inputIndex++] & 0x3f;
          pending[pendingIndex++] = ((byte1 & 0x1f) << 12) | (byte2 << 6) | byte3;
        } else if ((byte1 & 0xf8) === 0xf0) {  // 4-byte
          const byte2 = bytes[inputIndex++] & 0x3f;
          const byte3 = bytes[inputIndex++] & 0x3f;
          const byte4 = bytes[inputIndex++] & 0x3f;
    
          // this can be > 0xffff, so possibly generate surrogates
          let codepoint = ((byte1 & 0x07) << 0x12) | (byte2 << 0x0c) | (byte3 << 0x06) | byte4;
          if (codepoint > 0xffff) {
            // codepoint &= ~0x10000;
            codepoint -= 0x10000;
            pending[pendingIndex++] = (codepoint >>> 10) & 0x3ff | 0xd800;
            codepoint = 0xdc00 | codepoint & 0x3ff;
          }
          pending[pendingIndex++] = codepoint;
        } else {
          // invalid initial byte
        }
      }
    }
    
    // Decoding a string is pretty slow, but use alternative options where possible.
    let decodeImpl = decodeFallback;
    if (typeof Buffer === 'function' && Buffer.from) {
      // Buffer.from was added in Node v5.10.0 (2015-11-17).
      decodeImpl = decodeBuffer;
    } else if (typeof Blob === 'function' && typeof URL === 'function' && typeof URL.createObjectURL === 'function') {
      // Blob and URL.createObjectURL are available from IE10, Safari 6, Chrome 19
      // (all released in 2012), Firefox 19 (2013), ...
      decodeImpl = decodeSyncXHR;
    }
    
    /**
     * @param {(!ArrayBuffer|!ArrayBufferView)} buffer
     * @param {{stream: boolean}=} options
     * @return {string}
     */
    FastTextDecoder.prototype['decode'] = function(buffer, options) { // (buffer, options={stream: false}) {
      if (options['stream']) {
        //throw new Error(`Failed to decode: the 'stream' option is unsupported.`);
      }
    
      let bytes;
    
      if (buffer instanceof Uint8Array) {
        // Accept Uint8Array instances as-is.
        bytes = buffer;
      } else if (buffer.buffer instanceof ArrayBuffer) {
        // Look for ArrayBufferView, which isn't a real type, but basically
        // represents all the valid TypedArray types plus DataView. They all have
        // ".buffer" as an instance of ArrayBuffer.
        bytes = new Uint8Array(buffer.buffer);
      } else {
        // The only other valid argument here is that "buffer" is an ArrayBuffer.
        // We also try to convert anything else passed to a Uint8Array, as this
        // catches anything that's array-like. Native code would throw here.
        bytes = new Uint8Array(buffer);
      }
    
      return decodeImpl(/** @type {!Uint8Array} */ (bytes));
    }

    TextDecoder = FastTextDecoder;
    
    }(this));

/**
 * GBXjs - Version 2021-13-09
 *
 * by BigBang1112 & ThaumicTom
 * released under MIT license
 */
 var GBX = Object;
 (function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        define(factory);
    } else if (typeof exports === 'object') {
        module.exports = factory();
    } else {
        GBX = factory();
    }
})(this, function () {
    // Main

    var gbx = 0;
    var version = 0;
    var format = 0;
    var refTableCompression = 0;
    var bodyCompression = 0;
    var unknown = 0;
    var classID = 0;
    var userDataSize = 0;
    var numHeaderChunks = 0;
    var byte = 0;
    var bytes = 0;
    var buffer = 0;
    var err = 0;
    var pointer = 0;
    var lookbackVersion = 0;
    var metadata = 0;

    var headerChunks = [];
    var lookbackStrings = [];

    var utf8decoder = new TextDecoder();

    var collectionIDs = {
        6: 'Stadium',
        11: 'Valley',
        12: 'Canyon',
        13: 'Lagoon',
        25: 'Stadium256',
        26: 'Stadium',
        10003: 'Common',
    };

    // Functions

    function optionProcess(data, f) {
        if (data['thumbnail']) {
            thumb = data['thumbnail'];
        }

        onParse = data['onParse'];
        onThumb = data['onThumb'];
        this.buffer = data['data'];

        if (typeof this.buffer != 'undefined') {
            f(this.buffer);
        } else {
            err = 1;
        }
    }

    function readFile(file) {
        return function (res, rej) {
            let fr = new FileReader();
            fr.addEventListener('loadend', function (e) {res(e.target.result)});
            fr.addEventListener('error', rej);
            fr.readAsArrayBuffer(file);
        };
    }

    function changeBuffer(newBuffer) {
        buffer = newBuffer;
        //var previousPointerPos = pointer;
        pointer = 0;
    }

    function peekByte() {
        return buffer[pointer];
    }

    /* Testing function
    
    function peekBytes(count) {
        var bytes = new Uint8Array(count);
        for (i = 0; i < count; i++)
            bytes[i] = buffer[pointer + i];
        return bytes;
    }
    */

    function readByte() {
        byte = peekByte();
        pointer += 1;
        return byte;
    }

    function readBytes(count) {
        bytes = new Uint8Array(count);
        for (i = 0; i < count; i++) bytes[i] = readByte();
        return bytes;
    }

    function readInt16() {
        bytes = readBytes(2);
        return bytes[0] | (bytes[1] << 8);
    }

    function readInt32() {
        bytes = readBytes(4);
        return bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    }

    function readInt64() {
        bytes = readBytes(8);
        return (
            bytes[0] |
            (bytes[1] << 8) |
            (bytes[2] << 16) |
            (bytes[3] << 24) |
            (bytes[4] << 32) |
            (bytes[5] << 40) |
            (bytes[6] << 48) |
            (bytes[7] << 56)
        );
    }

    function readString(count) {
        if (count == undefined) count = readInt32();
        return utf8decoder.decode(readBytes(count));
    }

    function readChar() {
        return readString(1);
    }

    function readLookbackString() {
        if (lookbackVersion == null) lookbackVersion = readInt32();

        var index = new Uint32Array([readInt32()])[0];

        if ((index & 0x3fff) == 0 && (index >> 30 == 1 || index >> 30 == -2)) {
            var str = readString();
            lookbackStrings.push(str);
            return str;
        } else if ((index & 0x3fff) == 0x3fff) {
            switch (index >> 30) {
                case 2:
                    return 'Unassigned';
                case 3:
                    return '';
                default:
                    err = 5;
            }
        } else if (index >> 30 == 0) {
            if (collectionIDs[index] == undefined) {
                err = 10;
                return index;
            } else return collectionIDs[index];
        } else if (lookbackStrings.Count > (index & 0x3fff) - 1)
            return lookbackStrings[(index & 0x3fff) - 1];
        else return '';
    }

    function readMeta() {
        return {
            id: readLookbackString(),
            collection: readLookbackString(),
            author: readLookbackString(),
        };
    }

    function readBool() {
        return !!readInt32();
    }

    function readFloat() {
        return readInt32().toFixed(2);
    }

    function readVec2() {
        return {
            x: readFloat(),
            y: readFloat(),
        };
    }

    function toBase64(data) {
        return btoa(
            new Uint8Array(data).reduce(function (data, byte) {
                return data + String.fromCharCode(byte);
            }, '')
        );
    }

    function deformat(str) {
        return str.replace(/(\$)\$|\$([a-f0-9]{2,3}|[lh]\[.*?\]|.)/gi, '$1');
    }

    function getGameByTitleUID(title) {
        if (typeof title == 'undefined') {
            return 'Trackmania 1';
        } else if (title == 'Trackmania' || title.match(/^OrbitalDev@/g)) {
            return 'Trackmania';
        } else if (title == 'TMCE@nadeolabs' || title == 'TMTurbo@nadeolabs') {
            return 'Trackmania Turbo';
        } else {
            return 'ManiaPlanet';
        }
    }

    // Process options; Starting point

    function GBX(data) {
        f = GBX.read;

        if (data['data'].constructor !== Uint8Array && !data['force']) {
            (function () {
                data['data'] = new Uint8Array(readFile(data['data']));
                optionProcess(data, f);
            })();
        } else {
            optionProcess(data, f);
        }
    }

    GBX.prototype.read = function (buffer) {
        metadata = [];
        headerChunks = [];

        pointer = 0;
        err = 0;
        lookbackVersion = null;

        buffer = this.buffer;

        gbx = readString(3);

        if (gbx == 'GBX') {
            version = readInt16();

            if (version >= 3) {
                format = readChar();
                refTableCompression = readChar();
                bodyCompression = readChar();
                if (version >= 4) unknown = readChar();
                classID = readInt32();

                if (version >= 6) {
                    userDataSize = readInt32();
                    if (userDataSize > 0) {
                        numHeaderChunks = readInt32();

                        for (a = 0; a < numHeaderChunks; a++) {
                            var chunkId = readInt32() & 0xfff;
                            var chunkSize = readInt32();
                            var isHeavy = (chunkSize & (1 << 31)) != 0;

                            headerChunks[chunkId] = {
                                size: chunkSize & ~0x80000000,
                                isHeavy: isHeavy,
                            };
                        }

                        for (var key in headerChunks) {
                            headerChunks[key].data = readBytes(
                                headerChunks[key].size
                            );
                            delete headerChunks[key].size;
                        }

                        if (classID == 0x03043000 || classID == 0x24003000) {
                            // Map
                            metadata.type = 'Map';

                            changeBuffer(headerChunks[0x002].data);

                            var chunk002Version = readByte();
                            if (chunk002Version < 3) {
                                metadata.mapInfo = readMeta();
                                metadata.mapName = readString();
                            }
                            readInt32();
                            if (chunk002Version >= 1) {
                                metadata.bronzeTime = readInt32();
                                metadata.silverTime = readInt32();
                                metadata.goldTime = readInt32();
                                metadata.authorTime = readInt32();
                                if (chunk002Version == 2) readByte();
                                if (chunk002Version >= 4) {
                                    metadata.cost = readInt32();
                                    if (chunk002Version >= 5) {
                                        metadata.isMultilap = readBool();
                                        if (chunk002Version == 6) readBool();
                                        if (chunk002Version >= 7) {
                                            metadata.trackType = readInt32();
                                            if (chunk002Version >= 9) {
                                                readInt32();
                                                if (chunk002Version >= 10) {
                                                    metadata.authorScore =
                                                        readInt32();
                                                    if (chunk002Version >= 11) {
                                                        metadata.editorMode =
                                                            readInt32(); // bit 0: advanced/simple editor, bit 1: has ghost blocks
                                                        metadata.isSimple =
                                                            (metadata.editorMode &
                                                                1) !=
                                                            0;
                                                        metadata.hasGhostBlocks =
                                                            (metadata.editorMode &
                                                                2) !=
                                                            0;
                                                        if (
                                                            chunk002Version >=
                                                            12
                                                        ) {
                                                            readBool();
                                                            if (
                                                                chunk002Version >=
                                                                13
                                                            ) {
                                                                metadata.nbCheckpoints =
                                                                    readInt32();
                                                                metadata.nbLaps =
                                                                    readInt32();
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            changeBuffer(headerChunks[0x003].data); // Change to 0x03043003 (Common)

                            var chunk003Version = readByte();
                            metadata.mapInfo = readMeta();
                            metadata.mapName = readString();
                            metadata.mapNameD = deformat(metadata.mapName);
                            var kind = readByte();
                            if (kind == 6)
                                // Unvalidated map
                                err = 4;
                            if (chunk003Version >= 1) {
                                metadata.locked = readBool(); // used by Virtual Skipper to lock the map parameters
                                metadata.password = readString(); // weak xor encryption, no longer used in newer track files; see 03043029
                                if (chunk003Version >= 2) {
                                    metadata.decoration = readMeta();
                                    if (chunk003Version >= 3) {
                                        metadata.mapOrigin = readVec2();
                                        if (chunk003Version >= 4) {
                                            metadata.mapTarget = readVec2();
                                            if (chunk003Version >= 5) {
                                                readInt64();
                                                readInt64();
                                                if (chunk003Version >= 6) {
                                                    metadata.mapType =
                                                        readString();
                                                    metadata.mapStyle =
                                                        readString();
                                                    if (chunk003Version <= 8)
                                                        readBool();
                                                    if (chunk003Version >= 8) {
                                                        metadata.lightmapCacheUID =
                                                            readBytes(8);
                                                        if (
                                                            chunk003Version >= 9
                                                        ) {
                                                            metadata.lightmapVersion =
                                                                readByte();
                                                            if (
                                                                chunk003Version >=
                                                                11
                                                            )
                                                                metadata.titleUID =
                                                                    readLookbackString();
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            metadata.game = getGameByTitleUID(
                                metadata.titleUID
                            );

                            changeBuffer(headerChunks[0x005].data);

                            metadata.xml = readString();

                            if (chunk003Version > 5) {
                                changeBuffer(headerChunks[0x008].data);

                                /*var chunk008Version =*/
                                readInt32();
                                metadata.authorVersion = readInt32();
                                metadata.authorLogin = readString();
                                metadata.authorNickname = readString();
                                metadata.authorZone = readString();
                                metadata.authorExtraInfo = readString();
                            }
                        } else if (
                            classID == 0x03093000 ||
                            classID == 0x2407e000 ||
                            classID == 0x2403f000
                        ) {
                            // Replay
                            metadata.type = 'Replay';

                            changeBuffer(headerChunks[0x000].data);

                            chunk000Version = readInt32();
                            if (chunk000Version >= 2) {
                                metadata.mapInfo = readMeta();
                                metadata.time = readInt32();
                                metadata.driverNickname = readString();

                                if (chunk000Version >= 6) {
                                    metadata.driverLogin = readString();

                                    if (chunk000Version >= 8) {
                                        readByte();
                                        metadata.titleUID =
                                            readLookbackString();
                                    }
                                }
                            }

                            metadata.game = getGameByTitleUID(
                                metadata.titleUID
                            );

                            changeBuffer(headerChunks[0x001].data);

                            metadata.xml = readString();

                            try {
                                changeBuffer(headerChunks[0x002].data);
                                var chunk002Version = readInt32();
                                metadata.authorVersion = readInt32();
                                metadata.authorLogin = readString();
                                metadata.authorNickname = readString();
                                metadata.authorZone = readString();
                                metadata.authorExtraInfo = readString();
                            } catch (e) {}
                        } else err = 3;
                    }
                }
            }
        } else err = 2;

        if (typeof onParse != 'undefined' && onParse.length > 0) {
            onParse(metadata, err, headerChunks, classID);
        }

        // Thumbnail

        if (thumb) {
            changeBuffer(headerChunks[0x007].data);

            var chunk007Version = readInt32();
            if (chunk007Version != 0) {
                metadata.thumbnailSize = readInt32();
                a;
                readString('<Thumbnail.jpg>'.length);
                if (metadata.thumbnailSize == 0) {
                    metadata.thumbnail = null;
                } else {
                    metadata.thumbnail = readBytes(metadata.thumbnailSize);
                }
                readString('</Thumbnail.jpg>'.length);
                readString('<Comments>'.length);
                metadata.comments = readString();
                readString('</Comments>'.length);
            }

            if (thumb == 'base64') {
                metadata.thumbnail = toBase64(metadata.thumbnail);
            }
        }

        if (typeof onThumb != 'undefined' && onThumb.length > 0) {
            onThumb(
                metadata.thumbnail,
                metadata.thumbnailSize,
                headerChunks
            );
        }
    };

    return GBX;
});