from tree_sitter import Node as NodeTS

from backend.src.domain.schemas.treesitter_types import (
    CLASS_TYPES,
    COMMENT_TYPES,
    DEF,
    EXPR,
    EXPRESSION_TYPES,
    FUNCTION_TYPES,
    IMPORT,
    IMPORT_TYPES,
    LITERAL_TYPES,
    SIMPLE,
    SIMPLE_TYPES,
    Block,
    CaseClause,
    ClassNode,
    Docstring,
    ExceptHandler,
    ForStatement,
    FunctionNode,
    IfStatement,
    ImportNode,
    MatchStatement,
    Node,
    Parameter,
    Parameters,
    Range,
    SimpleNode,
    TryStatement,
    WhileStatement,
    WithItem,
    WithStatement,
)
from backend.src.services.processing.utils.parse_utils import (
    _extract_comments,
    _extract_first_text,
    _get_body_block,
    _make_id,
    _node_type_to_name,
    find_identifier,
)

CLAUSE_TYPES = {"else_clause", "elif_clause", "case_clause"}


def parse_class_definition(
    node: NodeTS,
    parent: Node,
    decorators: list[str] | None = None,
) -> ClassNode:
    """parse a class_definition node."""
    if decorators is None:
        decorators = []
    range_obj = Range.from_node(node)
    full_text = node.text.decode("utf-8") if node.text else ""

    # extract name, bases, decorators
    name_field = node.child_by_field_name("name")

    name = (
        name_field.text.decode("utf-8")
        if name_field is not None and name_field.text is not None
        else ""
    )

    bases_field = node.child_by_field_name("bases")
    superclasses: list[str] | None = None

    if bases_field is not None and bases_field.text is not None:
        superclasses = [
            c.text.decode("utf-8") if c.text else ""
            for c in bases_field.named_children
            if c.type in ("dotted_name", "identifier")
        ]

    # Use parent's path to build scope chain (e.g., "callgraph_service.CallGraphService")
    path = f"{parent.path}.{name}" if parent.path else f"{parent.identifier}.{name}"

    # find block and extract comments + docstring
    comments = _extract_comments(node)
    block = _get_body_block(node)

    if block:
        doc_and_str_content = _extract_first_text(block, index=0, parent=parent)

        doc = doc_and_str_content[0] if doc_and_str_content else None

        str_content = doc_and_str_content[1] if doc_and_str_content else ""
    else:
        doc = None
        str_content = ""

    # text = full class definition text
    text = full_text

    header = full_text

    if block is not None and full_text:
        for block_child in block.named_children:
            if block_child.text:
                idx = full_text.find(block_child.text.decode("utf-8"))
                if idx >= 0:
                    header = full_text[:idx]
                break

    #  temporary node for body parsing correct scope
    #  ensures children get identifiers like "callgraph_service.CallGraphService:_get_parser"
    body_parent = Node(
        identifier="",
        path=path,
        parent_id=parent.path if parent.path else parent.identifier,
        type="class_definition",
        range=range_obj,
        text="",
        children=[],
        comments=[],
        file_id=parent.file_id,
    )

    # parse body — use body_parent for proper identifier scoping
    body = _parse_block(block, parent=body_parent) if block else Block(statements=[])

    return ClassNode(
        identifier=_make_id(
            type_name="Class",
            name=name,
            parent=parent,
        ),
        text=text,
        name=name,
        superclasses=superclasses,
        docstring=doc,
        decorators=decorators,
        comments=comments,
        path=path,
        range=range_obj,
        parent_id=path,
        body=body,
        children=body.statements,  # populate children from body
        file_id=parent.file_id,
        header=header,
    )


def parse_function_definition(
    node: NodeTS,
    parent: Node,
    decorators: list[str] | None = None,
) -> FunctionNode:
    """parse a function_definition node."""
    range_obj = Range.from_node(node)
    full_text = node.text.decode("utf-8") if node.text else ""

    if decorators is None:
        decorators = []

    name_field = node.child_by_field_name("name")

    name = (
        name_field.text.decode("utf-8")
        if name_field is not None and name_field.text is not None
        else find_identifier(node)
    )

    param_node = next((i for i in node.named_children if i.type == "parameters"), None)

    params_list: list[Parameter] = []

    if not param_node:
        params = Parameters()
    else:
        splat_kwargs = next(
            (
                i
                for i in param_node.named_children
                if i.type == "dictionary_splat_pattern"
                if param_node
            ),
            None,
        )
        splat_args = next(
            (
                i
                for i in param_node.named_children
                if i.type == "list_splat_pattern"
                if param_node
            ),
            None,
        )
        for param_node in param_node.named_children:
            # Skip punctuation and non-parameter nodes
            if param_node.type in ("(", ",", ")"):
                continue

            # Handle simple identifier parameters (e.g., self, x)
            if param_node.type == "identifier":
                param_text = param_node.text.decode("utf-8") if param_node.text else ""
                if param_text:
                    params_list.append(Parameter(name=param_text))
            # Handle typed parameters (e.g., x: int)
            elif param_node.type == "typed_parameter":
                children = [n for n in param_node.named_children]
                identifier = next(
                    (
                        c.text.decode("utf-8")
                        for c in children
                        if c.type == "identifier" and c.text
                    ),
                    None,
                )
                type_annot = next(
                    (
                        c.text.decode("utf-8")
                        for c in children
                        if c.type == "type" and c.text
                    ),
                    None,
                )
                if identifier:
                    params_list.append(
                        Parameter(name=identifier, type_annotation=type_annot)
                    )
            # Handle default parameters (e.g., x=1)
            elif param_node.type == "default_parameter":
                children = [n for n in param_node.named_children]
                identifier = next(
                    (
                        c.text.decode("utf-8")
                        for c in children
                        if c.type == "identifier" and c.text
                    ),
                    None,
                )
                default_val = next(
                    (
                        c.text.decode("utf-8")
                        for c in children
                        if c.text and c.type not in ("=", "identifier")
                    ),
                    None,
                )
                if identifier:
                    params_list.append(
                        Parameter(name=identifier, default_value=default_val)
                    )
            # Handle typed default parameters (e.g., x: int = 1)
            elif param_node.type == "typed_default_parameter":
                children = [n for n in param_node.named_children]
                identifier = next(
                    (
                        c.text.decode("utf-8")
                        for c in children
                        if c.type == "identifier" and c.text
                    ),
                    None,
                )
                type_annot = next(
                    (
                        c.text.decode("utf-8")
                        for c in children
                        if c.type == "type" and c.text
                    ),
                    None,
                )
                default_val = next(
                    (
                        c.text.decode("utf-8")
                        for c in children
                        if c.text and c.type not in ("=", "identifier", "type")
                    ),
                    None,
                )
                if identifier:
                    params_list.append(
                        Parameter(
                            name=identifier,
                            type_annotation=type_annot,
                            default_value=default_val,
                        )
                    )

        params: Parameters = Parameters(
            params=params_list,
            accepts_args=splat_args is not None,
            accepts_kwargs=splat_kwargs is not None,
        )

    # extract return type annotation
    return_type: str | None = None

    annotations_field = node.child_by_field_name("return_type")

    if annotations_field is not None and annotations_field.text is not None:
        return_type = annotations_field.text.decode("utf-8")

    # is_async
    is_async = any(child.type == "async" for child in node.children)

    comments = _extract_comments(node)

    block = _get_body_block(node)

    docstring: Docstring | None = None
    first_child_text: str = ""
    if block is not None:
        result = _extract_first_text(block, index=0, parent=parent)
        if result is not None:
            docstring, first_child_text = result

        # header = full function text (def line only, sliced before block body begins)
    header = full_text

    if block is not None and full_text:
        for block_child in block.named_children:
            if block_child.text:
                idx = full_text.find(block_child.text.decode("utf-8"))
                if idx >= 0:
                    header = full_text[:idx]
                break

    # extracting decorator names from parent (decorated_definition)
    containing_class: str | None = None

    def _get_containing_class(node) -> str | None:
        """Get the containing class name from a method's parent chain."""
        if node is None:
            return None
        # Case 1: decorated_definition -> class_definition
        if node.type == "decorated_definition":
            grandparent = getattr(node, "parent", None)
            if grandparent is not None and grandparent.type == "class_definition":
                name_field = grandparent.child_by_field_name("name")
                if name_field and name_field.text:
                    return name_field.text.decode("utf-8")
        # Case 2: block -> class_definition (regular method)
        elif node.type == "block":
            grandparent = getattr(node, "parent", None)
            if grandparent is not None and grandparent.type == "class_definition":
                name_field = grandparent.child_by_field_name("name")
                if name_field and name_field.text:
                    return name_field.text.decode("utf-8")
        return None

    if node.parent is not None:
        # check for decorators on decorated_definition
        if node.parent.type == "decorated_definition":
            parent_node = node.parent
            for dec in parent_node.named_children:
                if dec.type == "decorator":
                    identifier_child = dec.child_by_field_name("name")
                    if (
                        identifier_child is not None
                        and identifier_child.text is not None
                    ):
                        decorators.append(identifier_child.text.decode("utf-8"))
            # check if containing class
            containing_class = _get_containing_class(parent_node)
        elif node.parent.type == "block":
            # Regular method inside class body
            containing_class = _get_containing_class(node.parent)

    # use parent's path to build scope chain (e.g., "callgraph_service._extract_nodes")
    path = f"{parent.path}.{name}" if parent.path else f"{parent.identifier}.{name}"

    # build a temporary node for body parsing with correct scope chain
    #  ensures children get identifiers like "callgraph_service.CallGraphService:_get_parser"
    body_parent = Node(
        identifier="",  # not used for id building, path is used
        path=path,
        parent_id=parent.path if parent.path else parent.identifier,
        type="function_definition",
        range=range_obj,
        text="",
        children=[],
        comments=[],
    )

    # parse body — use body_parent for proper identifier scoping
    body = _parse_block(block, parent=body_parent) if block else Block(statements=[])

    # Determine receiver (first parameter if self/cls)
    receiver: str | None = None
    if params.params and len(params.params) > 0:
        first_param = params.params[0].name
        if first_param in ("self", "cls"):
            receiver = first_param

    return FunctionNode(
        identifier=_make_id(
            type_name=node.type,
            name=name,
            parent=parent,
        ),
        text=full_text,
        name=name,
        parameters=params,
        return_type=return_type,
        body=body,
        children=body.statements,  # populate children from body
        is_async=is_async,
        decorators=decorators,
        containing_class=containing_class,
        receiver=receiver,
        docstring=docstring,
        comments=comments,
        type="function_definition",
        statement_type="",
        path=path,
        range=range_obj,
        parent_id=parent.identifier,
        file_id=parent.file_id,
        header=header,
    )


def _extract_decorators(node: NodeTS) -> list[str]:
    """extract decorator text from a decorated_definition node."""
    decorators: list[str] = []
    for child in node.named_children:
        if child.type == "decorator":
            decorators.append(child.text.decode("utf-8") if child.text else "")
    return decorators


def parse_decorated_definition(
    node: NodeTS,
    parent: Node,
) -> Node:
    """parse a decorated_definition — delegate to class or function parser with decorators."""
    decorators = _extract_decorators(node)

    for child in node.named_children:
        if child.type == DEF.CLASS:
            return parse_class_definition(
                child,
                parent=parent,
                decorators=decorators,
            )
        elif child.type == DEF.FUNCTION:
            return parse_function_definition(
                child,
                parent=parent,
                decorators=decorators,
            )
    # fallback: generic node
    return Node(
        identifier=_make_id(
            type_name="DefinitionNode",
            name=find_identifier(node=node),
            parent=parent,
        ),
        text=node.text.decode("utf-8") if node.text else "",
        type="decorated_definition",
        path=parent.path,
        range=Range.from_node(node),
        parent_id=parent.path,
        children=[],
        file_id=parent.file_id,
    )


def parse_comment(
    node: NodeTS,
    parent: Node,
) -> Node:
    """parse a comment node — return a Node with comment type."""
    row = node.start_point.row
    # check if inline (same row as any sibling)
    return Node(
        identifier=_make_id(type_name="Comment", name=None, parent=parent),
        text=node.text.decode("utf-8") if node.text else "",
        type="comment",
        path=f"{parent.path}.comment:{row}",
        range=Range.from_node(node),
        parent_id=parent.path,
        children=[],
        file_id=parent.file_id,
    )


# TYPED COMPOUND STATEMENT PARSERS


def _parse_block(
    block: NodeTS,
    parent: Node,
) -> Block:
    """parse a block node into a Block struct with typed children."""
    statements: list[Node] = []

    for child in block.named_children:
        if child.type in ("comment", "identifier"):
            continue  # skip comments (collected by _extract_comments) and identifiers (used for naming)
        if child.type == "expression_statement":
            # expression_statement is a no-op container — add its children directly
            for sub_child in child.named_children:
                if sub_child.type in ("comment", "identifier"):
                    continue
                child_tree = parse_node(
                    sub_child,
                    parent=parent,
                )
                if child_tree is not None:
                    statements.append(child_tree)
            continue
        child_tree = parse_node(
            child,
            parent=parent,
        )
        if child_tree is not None:
            statements.append(child_tree)
    return Block(statements=statements)


def _parse_else_clause(
    node: NodeTS | None,
    parent: Node,
) -> Block | None:
    """parse an else_clause into a Block (not a separate node)."""
    if node is None:
        return None
    block = _get_body_block(node)
    if block is None:
        return None
    return _parse_block(block, parent=parent)


def _parse_elif_clause(
    node: NodeTS | None,
    parent: Node,
) -> Block | None:
    """parse an elif_clause into a Block (not a separate node)."""
    if node is None:
        return None
    block = _get_body_block(node)
    if block is None:
        return None
    return _parse_block(block, parent=parent)


def _parse_except_clause(
    node: NodeTS,
    parent: Node,
) -> ExceptHandler:
    """parse an except_clause into an ExceptHandler struct."""
    # get exception type
    exception_type: str | None = None
    alias: str | None = None

    # find the exception type (e.g. 'ValueError' in 'except ValueError as e')
    for child in node.named_children:
        if child.type == ("expression"):
            exception_type = child.text.decode("utf-8") if child.text else ""
            break

    # find the alias (e.g. 'e' in 'except ValueError as e')
    as_pattern = node.child_by_field_name("as_pattern")
    if as_pattern is not None and as_pattern.text is not None:
        alias = as_pattern.text.decode("utf-8")

    # parse the body block
    except_block = _get_body_block(node)
    if except_block is None:
        body = Block(statements=[])
    else:
        body = _parse_block(
            block=except_block,
            parent=parent,
        )

    return ExceptHandler(
        exception_type=exception_type,
        alias=alias,
        body=body,
    )


def parse_if_statement(
    node: NodeTS,
    parent: Node,
) -> IfStatement:
    """parse an if_statement — elif/else flow as fields, not separate nodes."""
    range_obj = Range.from_node(node)
    full_text = node.text.decode("utf-8") if node.text else ""

    # condition from the 'condition' field
    condition = ""
    condition_field = node.child_by_field_name("condition")
    if condition_field is not None and condition_field.text is not None:
        condition = condition_field.text.decode("utf-8")

    # header = text before the first body child
    header = full_text

    block = _get_body_block(node)

    # body
    body = Block(statements=[])

    if block is not None:
        if full_text:
            for block_child in block.named_children:
                if block_child.text:
                    idx = full_text.find(block_child.text.decode("utf-8"))
                    if idx >= 0:
                        header = full_text[:idx]
                    break

        body = _parse_block(block, parent=parent)

    # elif_clause (named child of if_statement)
    elif_body: Block | None = None
    for child in node.named_children:
        if child.type == "elif_clause":
            elif_body = _parse_elif_clause(
                node=child,
                parent=parent,
            )
            break

    # else_clause (named child of if_statement)
    else_body: Block | None = None
    for child in node.named_children:
        if child.type == "else_clause":
            else_body = _parse_else_clause(
                child,
                parent,
            )
            break

    # comments collected from the node's children
    comments = _extract_comments(node)

    return IfStatement(
        identifier=_make_id(
            type_name="IfStatement",
            name=None,
            parent=parent,
        ),
        text=header,
        condition=condition,
        body=body,
        elif_body=elif_body,
        else_body=else_body,
        comments=comments,
        type="if_statement",
        path=f"{parent.path}.if",  # parent will prepend module path
        range=range_obj,
        parent_id=parent.identifier,
        file_id=parent.file_id,
    )


def parse_for_statement(
    node: NodeTS,
    parent: Node,
) -> ForStatement:
    """parse a for_statement — else_clause flows as else_body field."""
    range_obj = Range.from_node(node)
    full_text = node.text.decode("utf-8") if node.text else ""

    # target and iterable from 'for_in_clause'
    target = ""
    iterable = ""
    for_in = node.child_by_field_name("left")  # for_in_clause

    if for_in is not None:
        for child in for_in.named_children:
            if child.type == "identifier":
                target = child.text.decode("utf-8") if child.text else ""
            elif child.type in (EXPR.CALL, EXPR.SUBSCRIPT, EXPR.ATTRIBUTE):
                iterable = child.text.decode("utf-8") if child.text else ""
            elif child.type in SIMPLE_TYPES:
                iterable = child.text.decode("utf-8") if child.text else ""
            elif child.text:
                iterable = child.text.decode("utf-8")

    # header
    header = full_text
    block = _get_body_block(node)
    if block is not None and full_text:
        for block_child in block.named_children:
            if block_child.text:
                idx = full_text.find(block_child.text.decode("utf-8"))
                if idx >= 0:
                    header = full_text[:idx]
                break

    # body
    body = _parse_block(block, parent=parent) if block is not None else None

    # else_clause
    else_body: Block | None = None
    for child in node.named_children:
        if child.type == "else_clause":
            else_body = _parse_else_clause(child, parent=parent)
            break

    # async check
    is_async = any(child.type == "async" for child in node.children)

    comments = _extract_comments(node)

    return ForStatement(
        identifier=_make_id(
            type_name="ForStatement",
            name=None,
            parent=parent,
        ),
        text=header,
        target=target,
        iterable=iterable,
        body=body,
        else_body=else_body,
        is_async=is_async,
        comments=comments,
        type="for_statement",
        path=f"{parent.path}.for",
        range=range_obj,
        parent_id=parent.identifier,
        file_id=parent.file_id,
    )


def parse_while_statement(
    node: NodeTS,
    parent: Node,
) -> WhileStatement:
    """parse a while_statement — else_clause flows as else_body field."""
    range_obj = Range.from_node(node)
    full_text = node.text.decode("utf-8") if node.text else ""

    # condition
    condition = ""
    condition_field = node.child_by_field_name("condition")
    if condition_field is not None and condition_field.text is not None:
        condition = condition_field.text.decode("utf-8")

    # header
    header = full_text
    block = _get_body_block(node)
    if block is not None and full_text:
        for block_child in block.named_children:
            if block_child.text:
                idx = full_text.find(block_child.text.decode("utf-8"))
                if idx >= 0:
                    header = full_text[:idx]
                break

    # body
    body = _parse_block(block, parent=parent) if block is not None else None

    # else_clause
    else_body: Block | None = None
    for child in node.named_children:
        if child.type == "else_clause":
            else_body = _parse_else_clause(
                child,
                parent=parent,
            )
            break

    comments = _extract_comments(node)

    return WhileStatement(
        identifier=_make_id(
            type_name="WhileStatement",
            name=None,
            parent=parent,
        ),
        text=header,
        condition=condition,
        body=body,
        else_body=else_body,
        comments=comments,
        type="while_statement",
        path=f"{parent.path}.while",
        range=range_obj,
        parent_id=parent.identifier,
    )


def parse_try_statement(
    node: NodeTS,
    parent: Node,
) -> TryStatement:
    """parse a try_statement — except/else/finally flow as fields."""
    range_obj = Range.from_node(node)
    full_text = node.text.decode("utf-8") if node.text else ""

    # body
    block = _get_body_block(node)
    body = (
        _parse_block(block, parent=parent)
        if block is not None
        else Block(statements=[])
    )

    # header
    header = full_text
    if block is not None and full_text:
        for block_child in block.named_children:
            if block_child.text:
                idx = full_text.find(block_child.text.decode("utf-8"))
                if idx >= 0:
                    header = full_text[:idx]
                break

    # except_clause(s) → handlers
    handlers: list[ExceptHandler] = []
    for child in node.named_children:
        if child.type == "except_clause":
            handlers.append(
                _parse_except_clause(
                    child,
                    parent=parent,
                )
            )

    # else_clause
    else_body: Block | None = None
    for child in node.named_children:
        if child.type == "else_clause":
            else_body = _parse_else_clause(
                child,
                parent=parent,
            )
            break

    # finally_clause
    finally_body: Block | None = None
    for child in node.named_children:
        if child.type == "finally_clause":
            finally_block = _get_body_block(child)
            finally_body = (
                _parse_block(
                    finally_block,
                    parent=parent,
                )
                if finally_block is not None
                else None
            )
            break

    comments = _extract_comments(node)

    return TryStatement(
        identifier=_make_id(
            type_name="TryStatement",
            name=None,
            parent=parent,
        ),
        text=header,
        body=body,
        handlers=handlers,
        else_body=else_body,
        finally_body=finally_body,
        comments=comments,
        type="try_statement",
        path=f"{parent.path}.try",
        range=range_obj,
        parent_id=parent.identifier,
        file_id=parent.file_id,
    )


def parse_with_statement(
    node: NodeTS,
    parent: Node,
) -> WithStatement:
    """parse a with_statement — with_items flow as typed field."""
    range_obj = Range.from_node(node)
    full_text = node.text.decode("utf-8") if node.text else ""

    # with_items
    items: list[WithItem] = []
    for child in node.named_children:
        if child.type == "with_item":
            # value
            value = ""
            value_field = child.child_by_field_name("value")
            if value_field is not None and value_field.text is not None:
                value = value_field.text.decode("utf-8")

            # alias
            alias: str | None = None
            as_pattern = child.child_by_field_name("as_pattern")
            if as_pattern is not None and as_pattern.text is not None:
                alias = as_pattern.text.decode("utf-8")

            items.append(WithItem(value=value, alias=alias))

    # body
    block = _get_body_block(node)
    body = (
        _parse_block(block, parent=parent)
        if block is not None
        else Block(statements=[])
    )

    # header
    header = full_text
    if block is not None and full_text:
        for block_child in block.named_children:
            if block_child.text:
                idx = full_text.find(block_child.text.decode("utf-8"))
                if idx >= 0:
                    header = full_text[:idx]
                break

    # async check
    is_async = any(child.type == "async" for child in node.children)

    comments = _extract_comments(node)

    return WithStatement(
        identifier=_make_id(type_name="WithStatement", name=None, parent=parent),
        text=header,
        items=items,
        body=body,
        is_async=is_async,
        comments=comments,
        type="with_statement",
        path=f"{parent.path}.with",
        range=range_obj,
        parent_id=parent.identifier,
        file_id=parent.file_id,
    )


def parse_match_statement(
    node: NodeTS,
    parent: Node,
) -> MatchStatement:
    """parse a match_statement — case_clauses flow as typed field."""
    range_obj = Range.from_node(node)
    full_text = node.text.decode("utf-8") if node.text else ""

    # subject
    subject = ""
    subject_field = node.child_by_field_name("value")
    if subject_field is not None and subject_field.text is not None:
        subject = subject_field.text.decode("utf-8")

    # header
    header = full_text
    block = _get_body_block(node)
    if block is not None and full_text:
        for block_child in block.named_children:
            if block_child.text:
                idx = full_text.find(block_child.text.decode("utf-8"))
                if idx >= 0:
                    header = full_text[:idx]
                break

    # case_clauses → cases
    cases: list[CaseClause] = []
    for child in node.named_children:
        if child.type == "case_clause":
            # pattern
            pattern = ""
            pattern_field = child.child_by_field_name("pattern")
            if pattern_field is not None and pattern_field.text is not None:
                pattern = pattern_field.text.decode("utf-8")

            # guard (optional)
            guard: str | None = None
            guard_field = child.child_by_field_name("guard")
            if guard_field is not None and guard_field.text is not None:
                guard = guard_field.text.decode("utf-8")

            # body
            case_block = _get_body_block(child)
            case_block = _get_body_block(child)
            body = (
                _parse_block(case_block, parent=parent)
                if case_block is not None
                else Block(statements=[])
            )

            cases.append(CaseClause(pattern=pattern, guard=guard, body=body))

    comments = _extract_comments(node)

    return MatchStatement(
        identifier=_make_id(
            type_name="MatchStatement",
            name=None,
            parent=parent,
        ),
        text=header,
        subject=subject,
        cases=cases,
        comments=comments,
        type="match_statement",
        path=f"{parent.path}.match",
        range=range_obj,
        parent_id=parent.identifier,
        file_id=parent.file_id,
    )


def _parse_aliased(child: NodeTS) -> tuple[str | None, str | None]:
    """extract dotted_name and alias from an aliased_import child."""
    dotted = child.child_by_field_name("dotted_name")
    alias = child.child_by_field_name("alias")
    dotted_name = (
        dotted.text.decode("utf-8")
        if dotted is not None and dotted.text is not None
        else None
    )
    alias_name = (
        alias.text.decode("utf-8")
        if alias is not None and alias.text is not None
        else None
    )
    return dotted_name, alias_name


def parse_import(
    node: NodeTS,
    parent_id: str,
    file_id: str,
) -> ImportNode:
    """parse an import statement node into an ImportNode with typed identifier."""
    range_obj = Range.from_node(node)

    text = node.text.decode("utf-8") if node.text else ""
    node_type = node.type

    from_dotted_names: list[str] = []
    import_dotted_names: list[str] = []
    aliased: str | None = None

    if node_type == IMPORT.FUTURE:
        import_type = IMPORT.FUTURE

        for child in node.children:
            if child.type == "dotted_name":
                import_dotted_names.append(
                    child.text.decode("utf-8") if child.text else ""
                )

                from_dotted_names.append("__future__")

    elif node_type == IMPORT.FROM:
        import_type = IMPORT.FROM

        for child in node.children:
            if child.type == "dotted_name":
                name = child.text.decode("utf-8") if child.text else ""
                # first dotted_name is the 'from' module, rest are imports
                if not from_dotted_names:
                    from_dotted_names = name.split(".")
                else:
                    import_dotted_names.append(name)
            elif child.type == "aliased_import":
                dotted_name, alias_name = _parse_aliased(child)
                if dotted_name is not None:
                    import_dotted_names.append(dotted_name)
                if alias_name is not None:
                    aliased = alias_name

    else:
        import_type = IMPORT.DEFAULT

        for child in node.children:
            if child.type == "dotted_name":
                import_dotted_names.append(
                    child.text.decode("utf-8") if child.text else ""
                )
            elif child.type == "aliased_import":
                dotted_name, alias_name = _parse_aliased(child)
                if dotted_name is not None:
                    import_dotted_names.append(dotted_name)
                if alias_name is not None:
                    aliased = alias_name

    # build identifier: ImportNode:from_module[import_names]
    from_str = ".".join(from_dotted_names) if from_dotted_names else ""
    imports_str = ", ".join(import_dotted_names) if import_dotted_names else ""

    identifier = (
        f"{parent_id}:{import_type}:{from_str}[{imports_str}]"
        if imports_str
        else f"{import_type}{from_str}"
    )

    # build path: module:module_name[identifier] for imports
    if parent_id:
        path = f"{parent_id}[{identifier}]"
    else:
        path = identifier

    return ImportNode(
        identifier=identifier,
        text=text,
        path=path,
        range=range_obj,
        parent_id=parent_id,
        from_dotted_names=from_dotted_names,
        import_dotted_names=import_dotted_names,
        aliased=aliased,
        type=import_type,
        statement=node.text.decode("utf-8") if node.text else "",
        file_id=file_id,
    )


def parse_simple_expressions(
    node: NodeTS,
    parent: Node,
) -> SimpleNode:
    """parse a simple statement node into a SimpleNode."""
    range_obj = Range.from_node(node)
    text = node.text.decode("utf-8") if node.text else ""
    node_type = node.type

    # if node is expression_statement, check children for actual SIMPLE type
    actual_node = node

    if node_type == "expression":
        for child in node.named_children:
            if child.type in (
                SIMPLE.ASSIGNMENT,
                SIMPLE.AUGMENTED_ASSIGNMENT,
                SIMPLE.YIELD,
            ):
                actual_node = child
                break

    statement_type = ""
    left: str | None = None
    value: str | None = None
    targets: list[str] = []
    names: list[str] = []

    if actual_node.type == SIMPLE.RETURN:
        statement_type = SIMPLE.RETURN
        for child in actual_node.named_children:
            if child.type == "identifier":
                value = child.text.decode("utf-8") if child.text else ""

    elif actual_node.type == SIMPLE.RAISE:
        statement_type = SIMPLE.RAISE
        for child in actual_node.named_children:
            value = child.text.decode("utf-8") if child.text else ""

    elif actual_node.type == SIMPLE.ASSERT:
        statement_type = SIMPLE.ASSERT
        for child in actual_node.named_children:
            value = child.text.decode("utf-8") if child.text else ""

    elif actual_node.type == SIMPLE.DELETE:
        statement_type = SIMPLE.DELETE
        for child in actual_node.named_children:
            targets.append(child.text.decode("utf-8") if child.text else "")

    elif actual_node.type == SIMPLE.ASSIGNMENT:
        statement_type = SIMPLE.ASSIGNMENT
        children = list(actual_node.named_children)
        if len(children) >= 1:
            left = children[0].text.decode("utf-8") if children[0].text else ""

    elif actual_node.type == SIMPLE.AUGMENTED_ASSIGNMENT:
        statement_type = SIMPLE.AUGMENTED_ASSIGNMENT
        left_field = actual_node.child_by_field_name("left")
        if left_field and left_field.text:
            left = left_field.text.decode("utf-8")

    elif actual_node.type == SIMPLE.TYPE_ALIAS:
        statement_type = SIMPLE.TYPE_ALIAS
        children = list(actual_node.named_children)
        if len(children) >= 1:
            left = children[0].text.decode("utf-8") if children[0].text else ""

    elif actual_node.type == SIMPLE.YIELD:
        statement_type = SIMPLE.YIELD
        for child in actual_node.named_children:
            value = child.text.decode("utf-8") if child.text else ""

    elif actual_node.type == SIMPLE.GLOBAL:
        statement_type = SIMPLE.GLOBAL
        for child in actual_node.named_children:
            names.append(child.text.decode("utf-8") if child.text else "")

    elif actual_node.type == SIMPLE.NONLOCAL:
        statement_type = SIMPLE.NONLOCAL
        for child in actual_node.named_children:
            names.append(child.text.decode("utf-8") if child.text else "")

    elif actual_node.type == SIMPLE.BREAK:
        statement_type = SIMPLE.BREAK

    elif actual_node.type == SIMPLE.CONTINUE:
        statement_type = SIMPLE.CONTINUE

    elif actual_node.type == SIMPLE.PASS:
        statement_type = SIMPLE.PASS

    elif actual_node.type == "expression":
        statement_type = "expression"
        for child in actual_node.named_children:
            if child.type not in ("assignment", "augmented_assignment", "yield"):
                value = child.text.decode("utf-8") if child.text else ""

    # build identifier: TypeName:name
    type_map: dict[str, str] = {
        SIMPLE.ASSIGNMENT: "AssignmentNode",
        SIMPLE.AUGMENTED_ASSIGNMENT: "AugmentedAssignmentNode",
        SIMPLE.RETURN: "ReturnStatement",
        SIMPLE.RAISE: "RaiseStatement",
        SIMPLE.ASSERT: "AssertStatement",
        SIMPLE.DELETE: "DeleteStatement",
        SIMPLE.TYPE_ALIAS: "TypeAliasStatement",
        SIMPLE.YIELD: "YieldExpression",
        SIMPLE.GLOBAL: "GlobalStatement",
        SIMPLE.NONLOCAL: "NonlocalStatement",
        SIMPLE.BREAK: "BreakStatement",
        SIMPLE.CONTINUE: "ContinueStatement",
        SIMPLE.PASS: "PassStatement",
        "expression": "ExpressionStatement",
    }
    type_name = type_map.get(statement_type, statement_type.title() + "Node")

    # pick the name: left for assignments, targets[0] for delete, names for global/nonlocal, value for others
    id_name = ""
    if left:
        id_name = left
    elif targets:
        id_name = targets[0]
    elif names:
        id_name = ",".join(names)
    elif value and statement_type in (
        SIMPLE.RETURN,
        SIMPLE.RAISE,
        SIMPLE.ASSERT,
        SIMPLE.YIELD,
    ):
        id_name = value

    identifier = _make_id(
        type_name=type_name,
        name=id_name,
        parent=parent,
    )

    if parent.path:
        path = f"{parent.identifier}[{identifier}]"
    else:
        path = f"{parent.identifier}{identifier}"

    return SimpleNode(
        identifier=identifier,
        text=text,
        path=path,
        range=range_obj,
        parent_id=parent.identifier,
        type=actual_node.type,
        statement_type=statement_type,
        file_id=parent.file_id,
    )


def _parse_expression_node(
    node: NodeTS,
    parent: Node,
) -> Node:
    """parse expression nodes (call, attribute, subscript, etc.) — text only, no children."""
    range_obj = Range.from_node(node)
    text = node.text.decode("utf-8") if node.text else ""

    node_type = node.type

    # map expression type to readable name
    type_name = _node_type_to_name(node_type)

    node_id = _make_id(
        type_name=type_name,
        name=find_identifier(node),
        parent=parent,
    )

    path = f"{parent.path}.{node_type}" if parent.path else node_type

    return Node(
        identifier=node_id,
        text=text,
        type=node_type,
        path=path,
        range=range_obj,
        parent_id=parent.path,
        children=[],  # expressions are leaf nodes — no recursion
        file_id=parent.file_id,
    )


def parse_node(
    node: NodeTS,
    parent: Node,
) -> Node | None:
    """dispatch to the appropriate parser based on node type."""
    node_type = node.type

    range_obj = Range.from_node(node)
    # breakpoint()
    text = node.text.decode("utf-8") if node.text else ""

    # dispatch by node type
    if node_type in IMPORT_TYPES:
        return parse_import(node, parent_id=parent.identifier, file_id=parent.file_id)

    elif node_type in CLASS_TYPES:
        return parse_class_definition(
            node=node,
            parent=parent,
        )

    elif node_type == "decorated_definition":
        return parse_decorated_definition(
            node=node,
            parent=parent,
        )

    elif node_type in FUNCTION_TYPES:
        return parse_function_definition(
            node=node,
            parent=parent,
        )

    elif node_type in COMMENT_TYPES:
        return parse_comment(
            node=node,
            parent=parent,
        )

    elif node_type == "if_statement":
        return parse_if_statement(
            node=node,
            parent=parent,
        )

    elif node_type == "for_statement":
        return parse_for_statement(
            node=node,
            parent=parent,
        )

    elif node_type == "while_statement":
        return parse_while_statement(
            node=node,
            parent=parent,
        )

    elif node_type == "try_statement":
        return parse_try_statement(
            node=node,
            parent=parent,
        )

    elif node_type == "with_statement":
        return parse_with_statement(
            node=node,
            parent=parent,
        )

    elif node_type == "match_statement":
        return parse_match_statement(
            node=node,
            parent=parent,
        )

    elif node_type in SIMPLE_TYPES:
        return parse_simple_expressions(
            node,
            parent=parent,
        )

    elif node_type in LITERAL_TYPES:
        # Literals are leaf text - skip, parent captures the text
        return None

    elif node_type in EXPRESSION_TYPES:
        # Expression nodes (call, attribute, subscript, etc.) - text only, no children
        return _parse_expression_node(node, parent=parent)

    elif node_type == "expression_statement":
        # Expression statement is a no-op container - parse children directly
        # Used at module level for module-level assignments
        for child in node.named_children:
            if child.type in ("comment", "identifier"):
                continue
            return parse_node(
                child,
                parent=parent,
            )

    # Fallback: clause types (else_clause, elif_clause, case_clause) as text-only
    # These appear as bare nodes in parent scope and should not recurse into children
    COMPOUND_TYPES = {"else_clause", "elif_clause", "case_clause"}

    identifier = find_identifier(node)

    path = f"{parent.identifier}.{identifier}"

    # For clause types, return as-is with no children (text-only)
    if node_type in COMPOUND_TYPES:
        return Node(
            identifier=_make_id(
                type_name=_node_type_to_name(node_type),
                name=identifier,
                parent=parent,
            ),
            text=text,
            type=node_type,
            path=path,
            range=range_obj,
            parent_id=parent.identifier,
            children=[],
            file_id=parent.file_id,
        )

    # For other fallback types, return as-is (no children recursion)
    return Node(
        identifier=_make_id(
            type_name=node.type,
            name=identifier,
            parent=parent,
        ),
        text=text,
        type=node_type,
        path=path,
        range=range_obj,
        parent_id=parent.identifier,
        children=[],
        file_id=parent.file_id,
    )
