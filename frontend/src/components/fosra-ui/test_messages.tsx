const messages_test = {
  conversation_id: "01JNW4X0000000000000000000",
  messages: [
    {
      message_id: "01JNW4X0000000000000000001",
      convo_id: "01JNW4X0000000000000000000",
      user_id: "01JNW4WZ000000000000000001",
      parent_id: null,
      role: "user",
      text: "What's the capital of France?",
      attached_files: [],
      attached_sources: [],
      created_at: "2026-02-11T10:00:00.000Z",
      message_metadata: {},
    },
    {
      message_id: "01JNW4X0500000000000000002",
      convo_id: "01JNW4X0000000000000000000",
      user_id: null,
      parent_id: "01JNW4X0000000000000000001",
      role: "assistant",
      text: "The capital of France is Paris.",
      attached_files: [],
      attached_sources: [],
      created_at: "2026-02-11T10:00:02.000Z",
      message_metadata: {},
    },
    {
      message_id: "01JNW4X0A00000000000000003",
      convo_id: "01JNW4X0000000000000000000",
      user_id: null,
      parent_id: "01JNW4X0000000000000000001",
      role: "assistant",
      text: "Paris is the capital and largest city of France.",
      attached_files: [],
      attached_sources: [
        {
          title: "Paris - Wikipedia",
          url: "https://en.wikipedia.org/wiki/Paris",
        },
      ],
      created_at: "2026-02-11T10:00:05.000Z",
      message_metadata: {
        branch_reason: "added_sources",
      },
    },
    {
      message_id: "01JNW4X0F00000000000000004",
      convo_id: "01JNW4X0000000000000000000",
      user_id: "01JNW4WZ000000000000000001",
      parent_id: "01JNW4X0500000000000000002",
      role: "user",
      text: "Tell me more about it.",
      attached_files: [],
      attached_sources: [],
      created_at: "2026-02-11T10:00:08.000Z",
      message_metadata: {},
    },
    {
      message_id: "01JNW4X0K00000000000000005",
      convo_id: "01JNW4X0000000000000000000",
      user_id: null,
      parent_id: "01JNW4X0F00000000000000004",
      role: "assistant",
      text: "Paris is known for the Eiffel Tower, the Louvre Museum, and its rich history.",
      attached_files: [],
      attached_sources: [],
      created_at: "2026-02-11T10:00:10.000Z",
      message_metadata: {},
    },
    {
      message_id: "01JNW4X0P00000000000000006",
      convo_id: "01JNW4X0000000000000000000",
      user_id: "01JNW4WZ000000000000000001",
      parent_id: "01JNW4X0A00000000000000003",
      role: "user",
      text: "What about its population?",
      attached_files: [],
      attached_sources: [],
      created_at: "2026-02-11T10:00:12.000Z",
      message_metadata: {},
    },
    {
      message_id: "01JNW4X0U00000000000000007",
      convo_id: "01JNW4X0000000000000000000",
      user_id: null,
      parent_id: "01JNW4X0P00000000000000006",
      role: "assistant",
      text: "Paris has approximately 2.1 million residents in the city proper.",
      attached_files: [],
      attached_sources: [],
      created_at: "2026-02-11T10:00:14.000Z",
      message_metadata: {},
    },
    {
      message_id: "01JNW4X0Z00000000000000008",
      convo_id: "01JNW4X0000000000000000000",
      user_id: null,
      parent_id: "01JNW4X0P00000000000000006",
      role: "assistant",
      text: "The city of Paris has about 2.1 million people, while the metropolitan area has over 12 million.",
      attached_files: [],
      attached_sources: [
        {
          title: "Demographics of Paris",
          url: "https://en.wikipedia.org/wiki/Demographics_of_Paris",
        },
      ],
      created_at: "2026-02-11T10:00:18.000Z",
      message_metadata: {
        branch_reason: "more_detailed",
      },
    },
    {
      message_id: "01JNW4X1400000000000000009",
      convo_id: "01JNW4X0000000000000000000",
      user_id: "01JNW4WZ000000000000000001",
      parent_id: "01JNW4X0K00000000000000005",
      role: "user",
      text: "When was the Eiffel Tower built?",
      attached_files: [],
      attached_sources: [],
      created_at: "2026-02-11T10:00:20.000Z",
      message_metadata: {},
    },
    {
      message_id: "01JNW4X1900000000000000010",
      convo_id: "01JNW4X0000000000000000000",
      user_id: null,
      parent_id: "01JNW4X1400000000000000009",
      role: "assistant",
      text: "The Eiffel Tower was built in 1889 for the World's Fair.",
      attached_files: [],
      attached_sources: [],
      created_at: "2026-02-11T10:00:22.000Z",
      message_metadata: {},
    },
    {
      message_id: "01JNW4X1E00000000000000011",
      convo_id: "01JNW4X0000000000000000000",
      user_id: null,
      parent_id: "01JNW4X1400000000000000009",
      role: "assistant",
      text: "The Eiffel Tower was constructed between 1887 and 1889 for the 1889 World's Fair, celebrating the 100th anniversary of the French Revolution.",
      attached_files: [],
      attached_sources: [
        {
          title: "Eiffel Tower - History",
          url: "https://www.toureiffel.paris/en/the-monument/history",
        },
      ],
      created_at: "2026-02-11T10:00:25.000Z",
      message_metadata: {
        branch_reason: "more_historical_detail",
      },
    },
  ],
};

// console.log(messages_test.messages[0]);
//

// console.log(Object.values(messages_test.messages.sort()));

const msgs = messages_test.messages;

// const renderFrom = (targetMsg) => {
// };

// target message id
// target parent id
//
// hold variable
//
// find message - message_id == target.parent_id
// hold variable == new message
// append new message to list
// ! loop
// find message - message_id === hold.parent_id
// append new message

function getAncestors(messages: typeof msgs) {
  const lastMessage = msgs[msgs.length - 1];
  var targetMsg = messages.find((i) => i.message_id === lastMessage.parent_id);

  const allAncestors: (typeof targetMsg)[] = [];

  while (true) {
    // console.log("loop");
    const ancestors: (typeof targetMsg)[] = msgs.filter(
      (i) => i.message_id === targetMsg?.parent_id,
    );

    // console.log(ancestors);

    allAncestors.push(ancestors);

    if (ancestors.at(0)?.parent_id === null) {
      console.log("entered");

      return allAncestors.reverse();
    }

    targetMsg = ancestors[ancestors.length - 1];
  }
}

console.log(getAncestors(msgs));
