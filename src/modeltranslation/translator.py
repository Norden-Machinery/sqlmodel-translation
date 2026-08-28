from collections.abc import Callable, Iterator
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from types import UnionType
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from pydantic._internal._decorators import Decorator, FieldSerializerDecoratorInfo
from pydantic_core import PydanticUndefined
from sqlalchemy import Column
from sqlalchemy.orm import column_property
from sqlmodel import SQLModel

from .exceptions import ImproperlyConfiguredError


class TranslationOptions:
    """Base class for configuring the translation of SQLModel classes.

    This class defines which fields are translated,
    which translations are required and how to handle missing values.

    Examples:
        >>> from modeltranslation import TranslationOptions
        >>> class BookTranslationOptions(TranslationOptions):
        ...     fields = ("title",)
        ...     required_languages = ("en",)

    """

    fields: tuple[str, ...] = ()
    """Names of fields to translate.

    Example:
        `(`title`, `description`)`
    """

    fallback_languages: dict[str, tuple[str, ...]] | None = None
    """Languages to use when the current language is missing.

    Example:
        `('en', 'pl', 'de')`

    The fallbacks can be also specified with a dictionary. The default key is required.

    Example:
        ```
        {
          'default': ('en', 'pl', 'de'),
          'fr': 'es'
        }
        ```
    """
    fallback_values: dict[str, Any] | Any = None
    """The values to use if all fallback languages yielded no value.

    Example:
        `('No translation provided')`

    It's also possible to specify a fallback value for each field.

    Example:
        ```
        {
          'title': ('No translation'),
          'author': ('No translation provided')
        }
        ```

    """

    fallback_undefined: dict[str, Any] | None = None

    required_languages: dict[str, tuple[str, ...]] | tuple[str, ...] | None = None
    """The required translations for this class.

    This also affects the pydantic model and typehints.

    Example:
        `('en',)`

    The fallbacks can be also specified with a dictionary.
    This makes it possible to set the requirements per field.

    Example:
        ```
        {
          'en': ('title', 'author'),
          'default': ('title',)
        }
        ```
    The `default` key is required.
    For english, title and author are required. For all other languages only title is required.

    """


class Translator:
    """A translator object that manages translations for registered SQLModel classes."""

    def __init__(
        self,
        default_language: str,
        languages: tuple[str, ...],
        fallback_languages: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Construct a translator object.

        Args:
            default_language (str): The language to use if no language was set externally.

            languages (tuple[str, ...]): All supported languages i.e the translations you want to store.

            fallback_languages (dict[str, tuple[str, ...]] | None): Fallbacks for each language
                used when the active language is not in `languages`. An example:
                `{
                  'default': ('en', 'pl', 'de'),
                  'fr': 'es'
                }`.
                The default key is required.

        Raises:
            ImproperlyConfiguredError: If the configuration is internally inconsistent.

        """
        self._active_language: ContextVar[str] = ContextVar("current_locale", default=default_language)

        self._default_language: str = default_language

        self._languages: tuple[str, ...] = languages

        # fallbacks for untranslated languages
        self._fallback_languages: dict[str, tuple[str, ...]] = {"default": (self._default_language,)}
        if fallback_languages:
            self._fallback_languages = fallback_languages

        self._validate_translator_object()

    def get_languages(self) -> tuple[str, ...]:
        return self._languages

    def get_active_language(self) -> str:
        return self._active_language.get()

    def set_active_language(self, locale: str) -> None:
        self._active_language.set(locale)

    def get_default_language(self) -> str:
        return self._default_language

    def register(self, model: type[SQLModel]) -> Callable:
        """Register a SQLModel class for translations.

        This function returns a decorator that applies `TranslationOptions`
        to the given SQLModel class. After applying, the model
        will have translation accessors and metadata set up automatically.

        Args:
            model (SQLModel): the class to apply translations on.

        Raises:
            ImproperlyConfiguredError: If the translation options are inconsistent with the Translator.

        Examples:
            >>> from sqlmodel import SQLModel
            >>> from modeltranslation import Translator
            ...
            >>> class Book(SQLModel, table=True):
            ...     title: str
            ...
            >>> translator = Translator(
            ...     default_language="en",
            ...     languages=("en", "pl"))
            ...
            >>> @translator.register(Book)
            ... class BookTranslationOptions(TranslationOptions):
            ...     fields=('title',)

        """

        def decorator(options: TranslationOptions) -> None:
            self._replace_accessors(model, options)
            self._rebuild_model(model, options)

        return decorator

    def _replace_accessors(  # noqa: C901
        self, model: type[SQLModel], options: TranslationOptions
    ) -> type[SQLModel]:
        # check if TranslationOptions are valid before modifing model
        self._validate_translation_options(options)

        def locale_get_decorator(original_get_function: Callable) -> Callable:
            @wraps(original_get_function)
            def locale_function(
                model_self: type[SQLModel] | SQLModel, name: str, *args: tuple[Any, ...]
            ) -> Callable:
                # ignore private and not translated functions
                if name.startswith("_") or name not in options.fields:
                    return original_get_function(model_self, name, *args)

                active_language = self.get_active_language()

                if active_language in self._languages:
                    value = original_get_function(model_self, f"{name}_{active_language}")
                    if not self._is_null_value(name, value, options):
                        return value

                for fallback_language in self._fallbacks_generator(active_language, options):
                    value = original_get_function(model_self, f"{name}_{fallback_language}")
                    if not self._is_null_value(name, value, options):
                        return value

                # no fallback language yielded a value, try fallback values
                return self._fallback_value(name, options)

            return locale_function

        def locale_set_decorator(original_set_function: Callable) -> Callable:
            @wraps(original_set_function)
            def locale_function(model_self: type[SQLModel], name: str, value: Any) -> Callable:  # noqa: ANN401
                if name.startswith("_") or name not in options.fields:
                    return original_set_function(model_self, name, value)

                active_language = self.get_active_language()
                # if language is in translation use it, else use the default translator language
                if active_language in self._languages:
                    return original_set_function(model_self, f"{name}_{active_language}", value)
                return original_set_function(model_self, f"{name}_{self._default_language}", value)

            return locale_function

        def locale_class_get_decorator(original_get_function: Callable) -> Callable:
            @wraps(original_get_function)
            def locale_function(
                model_self: type[SQLModel] | SQLModel, name: str, *args: tuple[Any, ...]
            ) -> Callable:
                if name.startswith("_") or name not in options.fields:
                    return original_get_function(model_self, name, *args)

                active_language = self.get_active_language()

                if active_language in self._languages:
                    return original_get_function(model_self, f"{name}_{active_language}")

                for fallback_language in self._fallbacks_generator(active_language, options):
                    return original_get_function(model_self, f"{name}_{fallback_language}")

                return original_get_function(model_self, f"{name}_{self._default_language}")

            return locale_function

        model.__class__.__getattribute__ = locale_class_get_decorator(model.__class__.__getattribute__)
        model.__getattribute__ = locale_get_decorator(model.__getattribute__)
        model.__setattr__ = locale_set_decorator(model.__setattr__)
        return model

    def _rebuild_model(self, model: type[SQLModel], options: TranslationOptions) -> None:
        def _find_field_owner(search_model: type[SQLModel], field: str) -> type[SQLModel]:
            """Find which class in the hierarchy actually defines this field."""
            for base in search_model.__mro__:
                if base is BaseModel or base is SQLModel or base is object:
                    continue
                if field in getattr(base, '__annotations__', {}):
                    return base
            return search_model  # default if not found

        def make_serializer(field_name: str) -> Callable:
            def serial(self: type[SQLModel], _: Any) -> Any:  # noqa: ANN401
                return getattr(self, field_name)

            return serial

        for field in options.fields:
            field_owner = _find_field_owner(model, field)
            orig_type = model.__table__.columns[field].type  # pyright: ignore[reportAttributeAccessIssue]
            orig_annotation = field_owner.__annotations__[field]

            # change field to be Nullable
            model.__table__.columns[field].nullable = True  # pyright: ignore[reportAttributeAccessIssue]
            field_owner.__annotations__[field] = self._make_optional(orig_annotation)
            model.model_fields[field].annotation = field_owner.__annotations__[field]

            # create the serializer function
            serializer_func = make_serializer(field)
            serializer_name = f"_serialize_{field}"
            setattr(model, serializer_name, serializer_func)

            # create and register the Decorator
            decorator = Decorator(
                cls_ref=f"{model.__module__}.{model.__qualname__}:{id(model)}",
                cls_var_name=serializer_name,
                func=serializer_func,
                shim=None,
                info=FieldSerializerDecoratorInfo(
                    fields=(field,),
                    mode='plain',
                    return_type=PydanticUndefined,
                    when_used='json',
                    check_fields=None
                )
            )
            model.__pydantic_decorators__.field_serializers[serializer_name] = decorator

            for lang in self._languages:
                translation_field = f"{field}_{lang}"

                translation_annotation = (
                    orig_annotation
                    if self._is_required(lang, field, options)
                    else self._make_optional(orig_annotation)
                )

                # change model SQL Alchemy table
                column = Column(
                    translation_field, orig_type, nullable=(not self._is_required(lang, field, options))
                )

                model.__table__.append_column(column)  # pyright: ignore[reportAttributeAccessIssue]

                # change model Pydantic field
                pydantic_field = deepcopy(model.model_fields[field])
                pydantic_field.exclude = True
                pydantic_field.alias = translation_field
                pydantic_field.annotation = translation_annotation

                model.model_fields[translation_field] = pydantic_field
                model.__annotations__[translation_field] = translation_annotation

                field_owner.__annotations__[translation_field] = translation_annotation
                setattr(model, translation_field, column_property(column))

        model.model_rebuild(force=True)

    def _make_optional(self, typehint: Any) -> Any:  # noqa: ANN401
        """Wrap a type in Optional[] unless it's already optional."""
        origin = get_origin(typehint)
        # if origin is Union and type(None) in get_args(typehint):
        if origin is UnionType and type(None) in get_args(typehint):
            return typehint
        return typehint | None

    def _is_required(self, language: str, field: str, options: TranslationOptions) -> bool:
        if type(options.required_languages) is tuple:
            return language in options.required_languages

        if type(options.required_languages) is dict:
            if language in options.required_languages:
                return field in options.required_languages[language]
            if "default" in options.required_languages:
                return field in options.required_languages["default"]
        # required_languages in TranslationOptions is None
        return False

    def _is_null_value(self, field: str, value: Any, options: TranslationOptions) -> bool:  # noqa: ANN401
        # if translation defines custom fallback undefined value then check if value is eq to it
        if options.fallback_undefined is not None and field in options.fallback_undefined:
            return value is None or value == options.fallback_undefined[field]
        # else check if value is eq None
        return value is None

    def _fallbacks_generator(self, language: str, options: TranslationOptions) -> Iterator[str]:
        if options.fallback_languages is not None:
            yield from self._yield_fallbacks(language, options.fallback_languages)
        elif self._fallback_languages is not None:
            yield from self._yield_fallbacks(language, self._fallback_languages)

    def _yield_fallbacks(self, language: str, fallbacks: dict[str, tuple[str, ...]]) -> Iterator[str]:
        seen: set[str] = set()

        for fallback in fallbacks.get(language, ()):
            if fallback not in seen:
                seen.add(fallback)
                yield fallback

        for fallback in fallbacks.get("default", ()):
            if fallback != language and fallback not in seen:
                seen.add(fallback)
                yield fallback

    def _fallback_value(self, field: str, options: TranslationOptions) -> Any:  # noqa: ANN401
        if options.fallback_values is None:
            return None
        if type(options.fallback_values) is not dict:
            return options.fallback_values
        if field in options.fallback_values:
            return options.fallback_values[field]
        return None

    def _validate_translator_object(self) -> None:
        if self._languages is None:
            msg = "'languages' can not be None"
            raise ImproperlyConfiguredError(msg)
        if type(self._languages) is not tuple:
            msg = f"'languages' type is invalid {type(self._languages)}"
            raise ImproperlyConfiguredError(msg)

        if self._default_language is None:
            msg = "'default_language' can not be None"
            raise ImproperlyConfiguredError(msg)

        if self._default_language not in self._languages:
            msg = f"'{self._default_language}' used in 'defult_language' not in defined languages {self._languages}"  # noqa: E501
            raise ImproperlyConfiguredError(msg)

        self._validate_fallback_languages(self._fallback_languages)

    def _validate_translation_options(self, options: TranslationOptions) -> None:
        self._validate_fallback_languages(options.fallback_languages)

        if options.required_languages is None:
            return

        if type(options.required_languages) is tuple:
            for lang in options.required_languages:
                if lang not in self._languages:
                    msg = f"'{lang}' used in 'required_languages' not in defined languages {self._languages}"
                    raise ImproperlyConfiguredError(msg)
        elif type(options.required_languages) is dict:
            for lang in options.required_languages:
                if lang != "default" and lang not in self._languages:
                    msg = f"'{lang}' used in 'required_languages' not in defined languages {self._languages}"
                    raise ImproperlyConfiguredError(msg)
        else:
            msg = f"'required_languages' type is invalid {type(options.required_languages)}"
            raise ImproperlyConfiguredError(msg)

    def _validate_fallback_languages(self, fallback_languages: dict[str, tuple[str, ...]] | None) -> None:
        if fallback_languages is None:
            return

        if type(fallback_languages) is not dict:
            msg = f"'fallback_languages' type is invalid {type(fallback_languages)}"
            raise ImproperlyConfiguredError(msg)

        if "default" not in fallback_languages:
            msg = "missing 'default' key in 'fallback_languages'"
            raise ImproperlyConfiguredError(msg)

        for key, value in fallback_languages.items():
            # check if languages used as keys are defined inside Translator languages
            if key != "default" and key not in self._languages:
                msg = f"'{key}' used in 'fallback_languages' not in defined languages {self._languages}"
                raise ImproperlyConfiguredError(msg)

            # check if fallbacks are defined as tuples
            if type(value) is not tuple:
                msg = f"'{key}' fallback type is invalid {type(value)}"
                raise ImproperlyConfiguredError(msg)

            # check if language is not used as fallback to itself
            if key != "default" and key in value:
                msg = f"'{key}' used in 'fallback_languages' used as fallback to itself"
                raise ImproperlyConfiguredError(msg)

            # check if languages used as fallbacks are defined inside Translator languages
            for lang in value:
                if lang not in self._languages:
                    msg = f"'{lang}' used in 'fallback_languages' not in defined languages {self._languages}"
                    raise ImproperlyConfiguredError(msg)
