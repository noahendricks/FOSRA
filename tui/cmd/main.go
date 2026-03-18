package main

import (
	"fmt"
	"os"

	tea "charm.land/bubbletea/v2"
	"github.com/roccoluxe/fosra-tui/tui/internal/ui"
)

func main() {
	f, err := tea.LogToFile("debug.log", "debug")
	if err != nil {
		fmt.Println("fatal:", err)
		os.Exit(1)
	}

	defer f.Close()

	p := tea.NewProgram(
		ui.NewApp(),
	)

	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
}
