package ui

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/glamour/ansi"
)

const markdownMargin = 1

var ansiEscapePattern = regexp.MustCompile("\x1b\\[[0-9;]*m")

func boolPtr(b bool) *bool       { return &b }
func stringPtr(s string) *string { return &s }
func uintPtr(u uint) *uint       { return &u }

func newMarkdownRenderer(width int) *glamour.TermRenderer {
	if width < 20 {
		width = 20
	}

	r, _ := glamour.NewTermRenderer(
		glamour.WithStyles(markdownStyleConfig()),
		glamour.WithWordWrap(width),
		glamour.WithPreservedNewLines(),
	)

	return r
}

func renderMarkdown(content string, width int) string {
	return renderMarkdownWithBackground(content, width, colorBg)
}

func renderMarkdownWithBackground(content string, width int, background string) string {
	if strings.TrimSpace(content) == "" {
		return ""
	}

	renderer := newMarkdownRenderer(width)
	rendered, err := renderer.Render(content)
	if err != nil {
		return content
	}

	rendered = forceReplaceBackground(rendered, background)
	return strings.Trim(rendered, "\n")
}

func markdownStyleConfig() ansi.StyleConfig {
	return ansi.StyleConfig{
		Document: ansi.StyleBlock{
			StylePrimitive: ansi.StylePrimitive{
				BlockPrefix: "",
				BlockSuffix: "",
				Color:       stringPtr(colorFg),
			},
			Margin: uintPtr(markdownMargin),
		},
		BlockQuote: ansi.StyleBlock{
			StylePrimitive: ansi.StylePrimitive{
				Color:  stringPtr(colorFgDim),
				Italic: boolPtr(true),
				Prefix: "┃ ",
			},
			Indent: uintPtr(1),
		},
		List: ansi.StyleList{
			LevelIndent: markdownMargin,
			StyleBlock: ansi.StyleBlock{
				StylePrimitive: ansi.StylePrimitive{
					Color: stringPtr(colorFg),
				},
			},
		},
		Heading: ansi.StyleBlock{
			StylePrimitive: ansi.StylePrimitive{
				BlockSuffix: "\n",
				Color:       stringPtr(colorBlue),
				Bold:        boolPtr(true),
			},
		},
		H1: ansi.StyleBlock{StylePrimitive: ansi.StylePrimitive{Prefix: "# ", Color: stringPtr(colorBlue), Bold: boolPtr(true)}},
		H2: ansi.StyleBlock{StylePrimitive: ansi.StylePrimitive{Prefix: "## ", Color: stringPtr(colorBlue), Bold: boolPtr(true)}},
		H3: ansi.StyleBlock{StylePrimitive: ansi.StylePrimitive{Prefix: "### ", Color: stringPtr(colorBlue), Bold: boolPtr(true)}},
		H4: ansi.StyleBlock{StylePrimitive: ansi.StylePrimitive{Prefix: "#### ", Color: stringPtr(colorBlue), Bold: boolPtr(true)}},
		H5: ansi.StyleBlock{StylePrimitive: ansi.StylePrimitive{Prefix: "##### ", Color: stringPtr(colorBlue), Bold: boolPtr(true)}},
		H6: ansi.StyleBlock{StylePrimitive: ansi.StylePrimitive{Prefix: "###### ", Color: stringPtr(colorBlue), Bold: boolPtr(true)}},
		Strikethrough: ansi.StylePrimitive{
			CrossedOut: boolPtr(true),
			Color:      stringPtr(colorComment),
		},
		Emph: ansi.StylePrimitive{
			Color:  stringPtr(colorCyan),
			Italic: boolPtr(true),
		},
		Strong: ansi.StylePrimitive{
			Bold:  boolPtr(true),
			Color: stringPtr(colorFg),
		},
		HorizontalRule: ansi.StylePrimitive{
			Color:  stringPtr(colorBorder),
			Format: "\n─────────────────────────────────────────\n",
		},
		Item: ansi.StylePrimitive{
			BlockPrefix: "• ",
			Color:       stringPtr(colorFg),
		},
		Enumeration: ansi.StylePrimitive{
			BlockPrefix: ". ",
			Color:       stringPtr(colorYellow),
		},
		Task: ansi.StyleTask{
			StylePrimitive: ansi.StylePrimitive{},
			Ticked:         "[✓] ",
			Unticked:       "[ ] ",
		},
		Link: ansi.StylePrimitive{
			Color:     stringPtr(colorCyan),
			Underline: boolPtr(true),
		},
		LinkText: ansi.StylePrimitive{
			Color: stringPtr(colorBlue),
			Bold:  boolPtr(true),
		},
		Code: ansi.StyleBlock{
			StylePrimitive: ansi.StylePrimitive{
				Color: stringPtr(colorCyan),
			},
		},
		CodeBlock: ansi.StyleCodeBlock{
			StyleBlock: ansi.StyleBlock{
				StylePrimitive: ansi.StylePrimitive{
					Prefix: " ",
					Color:  stringPtr(colorFg),
				},
				Margin: uintPtr(markdownMargin),
			},
			Chroma: &ansi.Chroma{
				Text:                ansi.StylePrimitive{Color: stringPtr(colorFg)},
				Error:               ansi.StylePrimitive{Color: stringPtr(colorRed)},
				Comment:             ansi.StylePrimitive{Color: stringPtr(colorComment)},
				CommentPreproc:      ansi.StylePrimitive{Color: stringPtr(colorPurple)},
				Keyword:             ansi.StylePrimitive{Color: stringPtr(colorPurple)},
				KeywordReserved:     ansi.StylePrimitive{Color: stringPtr(colorPurple)},
				KeywordNamespace:    ansi.StylePrimitive{Color: stringPtr(colorPurple)},
				KeywordType:         ansi.StylePrimitive{Color: stringPtr(colorBlue)},
				Operator:            ansi.StylePrimitive{Color: stringPtr(colorBlue)},
				Punctuation:         ansi.StylePrimitive{Color: stringPtr(colorFgDim)},
				Name:                ansi.StylePrimitive{Color: stringPtr(colorFg)},
				NameBuiltin:         ansi.StylePrimitive{Color: stringPtr(colorCyan)},
				NameTag:             ansi.StylePrimitive{Color: stringPtr(colorPurple)},
				NameAttribute:       ansi.StylePrimitive{Color: stringPtr(colorCyan)},
				NameClass:           ansi.StylePrimitive{Color: stringPtr(colorBlue)},
				NameConstant:        ansi.StylePrimitive{Color: stringPtr(colorOrange)},
				NameDecorator:       ansi.StylePrimitive{Color: stringPtr(colorPurple)},
				NameFunction:        ansi.StylePrimitive{Color: stringPtr(colorBlue)},
				LiteralNumber:       ansi.StylePrimitive{Color: stringPtr(colorOrange)},
				LiteralString:       ansi.StylePrimitive{Color: stringPtr(colorGreen)},
				LiteralStringEscape: ansi.StylePrimitive{Color: stringPtr(colorYellow)},
				GenericDeleted:      ansi.StylePrimitive{Color: stringPtr(colorRed)},
				GenericEmph:         ansi.StylePrimitive{Color: stringPtr(colorCyan), Italic: boolPtr(true)},
				GenericInserted:     ansi.StylePrimitive{Color: stringPtr(colorGreen)},
				GenericStrong:       ansi.StylePrimitive{Color: stringPtr(colorFg), Bold: boolPtr(true)},
				GenericSubheading:   ansi.StylePrimitive{Color: stringPtr(colorBlue)},
			},
		},
		Table: ansi.StyleTable{
			StyleBlock: ansi.StyleBlock{
				StylePrimitive: ansi.StylePrimitive{
					BlockPrefix: "\n",
					BlockSuffix: "\n",
				},
			},
			CenterSeparator: stringPtr("┼"),
			ColumnSeparator: stringPtr("│"),
			RowSeparator:    stringPtr("─"),
		},
		DefinitionDescription: ansi.StylePrimitive{
			BlockPrefix: "\n ❯ ",
			Color:       stringPtr(colorBlue),
		},
		Text: ansi.StylePrimitive{
			Color: stringPtr(colorFg),
		},
		Paragraph: ansi.StyleBlock{
			StylePrimitive: ansi.StylePrimitive{
				Color: stringPtr(colorFg),
			},
		},
	}
}

func forceReplaceBackground(input, background string) string {
	r, g, b, ok := parseHexColor(background)
	if !ok {
		return input
	}

	newBg := fmt.Sprintf("48;2;%d;%d;%d", r, g, b)

	return ansiEscapePattern.ReplaceAllStringFunc(input, func(seq string) string {
		const prefixLen = 2
		const suffixLen = 1

		raw := seq
		start := prefixLen
		end := len(raw) - suffixLen

		var sb strings.Builder
		sb.Grow((end - start) + len(newBg) + 2)

		for i := start; i < end; {
			j := i
			for j < end && raw[j] != ';' {
				j++
			}
			token := raw[i:j]

			if token == "48" {
				k := j + 1
				if k < end {
					l := k
					for l < end && raw[l] != ';' {
						l++
					}
					next := raw[k:l]
					if next == "5" {
						m := l + 1
						for m < end && raw[m] != ';' {
							m++
						}
						i = m + 1
						continue
					}
					if next == "2" {
						m := l + 1
						for count := 0; count < 3 && m < end; count++ {
							for m < end && raw[m] != ';' {
								m++
							}
							m++
						}
						i = m
						continue
					}
				}
			}

			value, err := strconv.Atoi(token)
			keep := err != nil || ((value < 40 || value > 47) && (value < 100 || value > 107) && value != 49)
			if keep {
				if sb.Len() > 0 {
					sb.WriteByte(';')
				}
				sb.WriteString(token)
			}

			i = j + 1
		}

		if sb.Len() > 0 {
			sb.WriteByte(';')
		}
		sb.WriteString(newBg)

		return "\x1b[" + sb.String() + "m"
	})
}

func parseHexColor(value string) (uint8, uint8, uint8, bool) {
	value = strings.TrimSpace(strings.TrimPrefix(value, "#"))
	if len(value) != 6 {
		return 0, 0, 0, false
	}

	r, err := strconv.ParseUint(value[0:2], 16, 8)
	if err != nil {
		return 0, 0, 0, false
	}
	g, err := strconv.ParseUint(value[2:4], 16, 8)
	if err != nil {
		return 0, 0, 0, false
	}
	b, err := strconv.ParseUint(value[4:6], 16, 8)
	if err != nil {
		return 0, 0, 0, false
	}

	return uint8(r), uint8(g), uint8(b), true
}
